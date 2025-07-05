import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# ─── CPU THREAD TUNING ─────────────────────────────────────────────
import multiprocessing
num_cpus = multiprocessing.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(num_cpus)
os.environ["MKL_NUM_THREADS"] = str(num_cpus)
torch.set_num_threads(num_cpus)
torch.set_num_interop_threads(num_cpus)
print(f"🔧 Using {num_cpus} CPU threads for training")

# ─── MODEL ─────────────────────────────────────────────────────────
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

# Optional torch.compile (PyTorch 2+)
try:
    QuadSolverNN = torch.compile(QuadSolverNN)
    print("⚡ Model compiled with torch.compile()")
except Exception:
    print("⚠️ torch.compile() not supported, running in eager mode")

# ─── DATA ───────────────────────────────────────────────────────────
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

# ─── TRAINING ────────────────────────────────────────────────────────
def train():
    model = QuadSolverNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 💾 Try resuming
    if os.path.exists("quad_solver_latest.pth"):
        try:
            model.load_state_dict(torch.load("quad_solver_latest.pth"))
            print("✅ Resumed from latest checkpoint.")
        except:
            print("⚠️ Could not load latest checkpoint. Starting fresh.")

    # 📦 Dataset: make it B I G
    total_samples = 4_500_000  # 6+ hrs training
    batch_size = 2048
    max_epochs = 2000
    patience = 10

    print("🧪 Generating data...")
    X, y = generate_data(total_samples)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=2,
                              pin_memory=True)

    val_loader = DataLoader(TensorDataset(X_val, y_val),
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=2,
                            pin_memory=True)

    best_loss = float('inf')
    bad_epochs = 0

    try:
        for epoch in range(1, max_epochs + 1):
            t0 = time.time()
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

            # ✅ Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    val_loss += criterion(model(xb), yb).item() * xb.size(0)
            val_loss /= len(val_loader.dataset)

            # 🧠 Logging
            dt = time.time() - t0
            print(f"🧠 Epoch {epoch:04d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | Time: {dt:.2f}s")

            # 💾 Save latest every epoch
            torch.save(model.state_dict(), "quad_solver_latest.pth")

            if val_loss < best_loss:
                best_loss = val_loss
                bad_epochs = 0
                torch.save(model.state_dict(), "quad_solver_best.pth")
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"🛑 Early stopping at epoch {epoch}")
                    break

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user. Saving emergency checkpoint...")
        torch.save(model.state_dict(), "quad_solver_emergency.pth")
        print("💾 Saved as quad_solver_emergency.pth — you’re safe fam.")

import time
if __name__ == "__main__":

while True:
    train()  # runs training until early stop or keyboard interrupt
    print("⏸️ Taking a 5-minute break to chill the VPS...")
    time.sleep(5 * 60)  # 5 minutes

