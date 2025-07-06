import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import multiprocessing

# ─── CPU THREAD TUNING ─────────────────────────────────────────────
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
def train(max_minutes=30):
    model = QuadSolverNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    start_epoch = 1
    best_loss = float('inf')

    # 🧬 Resume emergency checkpoint if exists
    if os.path.exists("quad_solver_emergency.pth"):
        try:
            checkpoint = torch.load("quad_solver_emergency.pth")
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            if 'scheduler_state' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state'])
            start_epoch = checkpoint.get('epoch', 1)
            best_loss = checkpoint.get('val_loss', float('inf'))
            print(f"🆘 Resumed from emergency checkpoint at epoch {start_epoch}, val_loss {best_loss:.6f}")
        except Exception as e:
            print(f"⚠️ Failed to load emergency checkpoint: {e}")
    else:
        print("🚫 No emergency checkpoint found. Starting from scratch.")

    # Dataset setup
    total_samples = 4_500_000
    batch_size = 2048
    max_epochs = 2000
    patience = 10
    bad_epochs = 0

    print("🧪 Generating data...")
    X, y = generate_data(total_samples)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers = min(8, num_cpus),
                              pin_memory=True)

    val_loader = DataLoader(TensorDataset(X_val, y_val),
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers = min(8, num_cpus),
                            pin_memory=True)

    start_time = time.time()

    try:
        for epoch in range(start_epoch, max_epochs + 1):
            epoch_start = time.time()
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

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    val_loss += criterion(model(xb), yb).item() * xb.size(0)
            val_loss /= len(val_loader.dataset)

            scheduler.step(val_loss)

            dt = time.time() - epoch_start
            total_elapsed = time.time() - start_time
            current_lr = optimizer.param_groups[0]['lr']

            print(f"🧠 Epoch {epoch:04d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {current_lr:.2e} | Time: {dt:.2f}s")

            # Save latest checkpoint
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'val_loss': val_loss
            }, "quad_solver_latest.pth")

            if val_loss < best_loss:
    best_loss = val_loss
    bad_epochs = 0
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'val_loss': val_loss
    }, "quad_solver_best.pth")
    print(f"💾 New best model saved at epoch {epoch} with val_loss {val_loss:.6f}")
    
    # 🔍 Evaluate the newly saved best model right away
    print("🎯 Evaluating best model on fresh samples...")
    evaluate_model(model, n_samples=5)


            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"🛑 Early stopping at epoch {epoch}")
                    break

            if total_elapsed > max_minutes * 60:
                print(f"⏰ Stopping training after {max_minutes} minutes this session.")
                break

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted. Saving emergency checkpoint...")
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'val_loss': val_loss
        }, "quad_solver_emergency.pth")
        print("💾 Emergency checkpoint saved as quad_solver_emergency.pth")

def evaluate_model(n_samples=5, model_path="quad_solver_best.pth"):
    print("\n📊 Evaluating the best model on fresh examples...\n")
    model = QuadSolverNN()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    # Generate test data
    a = np.random.uniform(1, 10, n_samples)
    b = np.random.uniform(-20, 20, n_samples)
    c = np.random.uniform(-50, 50, n_samples)

    for ai, bi, ci in zip(a, b, c):
        disc = bi**2 - 4 * ai * ci
        if disc < 0:
            continue  # skip imaginary roots

        x1 = (-bi + np.sqrt(disc)) / (2 * ai)
        x2 = (-bi - np.sqrt(disc)) / (2 * ai)
        real_roots = sorted([x1, x2])

        # Normalize input same as training
        x = torch.tensor([[ai / 10, bi / 20, ci / 50]], dtype=torch.float32)
        with torch.no_grad():
            pred = model(x).squeeze().tolist()
        pred = sorted(pred)

        print(f"🧪 Eq: {ai:.2f}x² + {bi:.2f}x + {ci:.2f} = 0")
        print(f"✅ Real: {real_roots}")
        print(f"🤖 Pred: {pred}")
        print("-" * 40)


if __name__ == "__main__":
    while True:
        train(max_minutes=30)
        evaluate_model(n_samples=5)  # 👈 NEW!
        print("⏸️ Taking a 5-minute break to chill the VPS...")
        time.sleep(5 * 60)
