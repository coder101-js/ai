import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# 🧠 Neural Net with dropout for VPS stress test
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

# 🧪 Generate synthetic quadratic data
def generate_data(n):
    a_vals = np.random.uniform(1, 10, n)
    b_vals = np.random.uniform(-20, 20, n)
    c_vals = np.random.uniform(-50, 50, n)
    data, labels = [], []
    for a, b, c in zip(a_vals, b_vals, c_vals):
        disc = b**2 - 4*a*c
        if disc < 0: continue
        x1 = (-b + np.sqrt(disc)) / (2*a)
        x2 = (-b - np.sqrt(disc)) / (2*a)
        data.append([a/10, b/20, c/50])  # Normalize input
        labels.append(sorted([x1, x2]))
    return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

# 🏋️ Training loop
def train():
    # ⚙️ Configs
    total_samples = 250_000     # 🔥 max out VPS
    batch_size    = 2048
    max_epochs    = 500
    max_bad       = 5
    lr            = 0.001

    model     = QuadSolverNN()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5, verbose=True)

    # 📦 Load data
    print("🔄 Generating data...")
    X, y = generate_data(total_samples)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"🚀 Training starting with {len(X_train)} train samples and {len(X_val)} val samples")
    best_loss = float('inf')
    bad_epochs = 0

    try:
        for epoch in range(1, max_epochs + 1):
            start = time.time()

            # 🎓 Train
            model.train()
            train_loss = 0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * xb.size(0)
            train_loss /= len(train_loader.dataset)

            # 🧪 Validate
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    val_loss += criterion(model(xb), yb).item() * xb.size(0)
            val_loss /= len(val_loader.dataset)

            scheduler.step(val_loss)

            duration = time.time() - start
            print(f"🧠 Epoch {epoch:03d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | Time: {duration:.2f}s")

            # 🧠 Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                bad_epochs = 0
                torch.save(model.state_dict(), "quad_solver_best.pth")
            else:
                bad_epochs += 1
                if bad_epochs >= max_bad:
                    print(f"🛑 Early stop at epoch {epoch} — no improvement for {bad_epochs} epochs.")
                    break

    except KeyboardInterrupt:
        # 💾 Emergency save
        print("⚠️ Training interrupted! Saving emergency model...")
        torch.save(model.state_dict(), "quad_solver_emergency.pth")
        print("✅ Saved as quad_solver_emergency.pth")

if __name__ == "__main__":
    train()
