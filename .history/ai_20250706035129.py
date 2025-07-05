import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# ─── CPU TUNING ───────────────────────────────────────────────────────────────
# Use all available vCPUs for linear algebra
import multiprocessing
num_cpus = multiprocessing.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(num_cpus)
os.environ["MKL_NUM_THREADS"] = str(num_cpus)
torch.set_num_threads(num_cpus)
torch.set_num_interop_threads(num_cpus)
print(f"🔧 Using {num_cpus} CPU threads for training")

# ─── MODEL DEFINITION ─────────────────────────────────────────────────────────
class QuadSolverNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

# Compile for graph-mode speedups (PyTorch 2.0+)
try:
    QuadSolverNN = torch.compile(QuadSolverNN)
    print("⚡ Successfully compiled the model with torch.compile()")
except Exception:
    print("⚠️ torch.compile() not available, running in eager mode")

# ─── DATA GENERATION ──────────────────────────────────────────────────────────
def generate_data(n):
    a = np.random.uniform(1, 10, n)
    b = np.random.uniform(-20, 20, n)
    c = np.random.uniform(-50, 50, n)
    data, labels = [], []
    for ai, bi, ci in zip(a, b, c):
        disc = bi**2 - 4*ai*ci
        if disc < 0: continue
        x1 = (-bi + np.sqrt(disc)) / (2*ai)
        x2 = (-bi - np.sqrt(disc)) / (2*ai)
        data.append([ai/10, bi/20, ci/50])
        labels.append(sorted([x1, x2]))
    return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

# ─── TRAINING LOOP ────────────────────────────────────────────────────────────
def train():
    # Config
    total_samples = 300_000      # bump it up
    batch_size    = 2048         # fits in ~4GB RAM
    lr            = 1e-3
    max_epochs    = 500
    max_bad       = 5            # early stop patience

    # Model / Optim / Scheduler
    model     = QuadSolverNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                    patience=3,
                                                    factor=0.5,
                                                    verbose=True)

    # Data
    print("🔄 Generating data...")
    X, y = generate_data(total_samples)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42)

    train_ds = TensorDataset(X_train, y_train)
    val_ds   = TensorDataset(X_val,   y_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_cpus,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_cpus,
        pin_memory=True,
        persistent_workers=True
    )

    print(f"🚀 Training on {len(X_train)} samples, validating on {len(X_val)} samples")
    best_loss, bad_epochs = float('inf'), 0

    try:
        for epoch in range(1, max_epochs+1):
            t0 = time.time()

            # — Train —
            model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * xb.size(0)
            train_loss /= len(train_loader.dataset)

            # — Validate —
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    val_loss += criterion(model(xb), yb).item() * xb.size(0)
            val_loss /= len(val_loader.dataset)

            # LR scheduler
            scheduler.step(val_loss)

            # Log
            dt = time.time() - t0
            print(f"🧠 Epoch {epoch:03d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | Time: {dt:.2f}s")

            # Checkpoint & Early Stop
            if val_loss < best_loss:
                best_loss = val_loss
                bad_epochs = 0
                torch.save(model.state_dict(), "quad_solver_best.pth")
            else:
                bad_epochs += 1
                if bad_epochs >= max_bad:
                    print(f"🛑 Early stopping at epoch {epoch}, no improvement for {bad_epochs} epochs.")
                    break

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted! Saving emergency checkpoint...")
        torch.save(model.state_dict(), "quad_solver_emergency.pth")
        print("💾  Saved as quad_solver_emergency.pth — you’re safe fam.")

if __name__ == "__main__":
    train()
