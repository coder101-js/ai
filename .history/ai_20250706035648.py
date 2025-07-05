import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# ─── CPU TUNING ─────────────────────────────────────
import multiprocessing
num_cpus = multiprocessing.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(num_cpus)
os.environ["MKL_NUM_THREADS"] = str(num_cpus)
torch.set_num_threads(num_cpus)
torch.set_num_interop_threads(num_cpus)
print(f"🔧 Using {num_cpus} CPU threads for training")

# ─── MODEL ───────────────────────────────────────────
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

# Optional torch.compile (skip if not supported)
try:
    QuadSolverNN = torch.compile(QuadSolverNN)
    print("⚡ Successfully compiled the model with torch.compile()")
except Exception:
    print("⚠️ torch.compile() not available, using eager mode.")

# ─── DATA ────────────────────────────────────────────
def generate_data(n):
    a = np.random.uniform(1, 10, n)
    b = np.random.uniform(-20, 20, n)
    c = np.random.uniform(-50, 50, n)
    data, labels = [], []
    for ai, bi, ci in zip(a, b, c):
        disc = bi**2 - 4 * ai * ci
        if disc < 0: continue
        x1 = (-bi + np.sqrt(disc)) / (2 * ai)
        x2 = (-bi - np.sqrt(disc)) / (2 * ai)
        data.append([ai / 10, bi / 20, ci / 50])  # scaling
        labels.append(sorted([x1, x2]))
    return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

# ─── TRAIN ───────────────────────────────────────────
def train():
    total_samples = 300_000
    batch_size = 2048
    lr = 1e-3
    max_epochs = 500
    max_bad = 5

    model = QuadSolverNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)  # no 'verbose'

    print("🔄 Generating data...")
    X, y = generate_data(total_samples)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_cpus, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_cpus, pin_memory=True)

    print(f"🚀 Training on {len(X_train)} samples, validating on {len(X_val)} samples")
    best_loss = float("inf")
    bad_epochs = 0

    try:
        for epoch in range(1, max_epochs + 1):
            start_time = time.time()

            model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * xb.size(0)
            train_loss /= len(train_loader.dataset)

            # 🔍 Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    val_loss += criterion(model(xb), yb).item() * xb.size(0)
            val_loss /= len(val_loader.dataset)

            scheduler.step(val_loss)

            elapsed = time.time() - start_time
            print(f"🧠 Epoch {epoch:03d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | Time: {elapsed:.2f}s")

            if val_loss < best_loss:
                best_loss = val_loss
                bad_epochs = 0
                torch.save(model.state_dict(), "quad_solver_best.pth")
            else:
                bad_epochs += 1
                if bad_epochs >= max_bad:
                    print(f"🛑 Early stopping at epoch {epoch}, no improvement for {max_bad} epochs.")
                    break

    except KeyboardInterrupt:
        print("\n🛑 CTRL+C detected. Saving emergency checkpoint...")
        torch.save(model.state_dict(), "quad_solver_emergency.pth")
        print("💾 Saved emergency checkpoint to quad_solver_emergency.pth.")

if __name__ == "__main__":
    train()
