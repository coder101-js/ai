import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
import time

# 🚀 Buffed model
class QuadSolverNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

# 🧠 Bigger data generator
def generate_data(n):
    data = []
    labels = []
    while len(data) < n:
        a = np.random.uniform(1, 100)
        b = np.random.uniform(-200, 200)
        c = np.random.uniform(-500, 500)
        disc = b**2 - 4*a*c
        if disc < 0: continue  # skip complex roots
        x1 = (-b + np.sqrt(disc)) / (2*a)
        x2 = (-b - np.sqrt(disc)) / (2*a)
        data.append([a / 100, b / 200, c / 500])  # normalize input
        labels.append(sorted([x1, x2]))  # sort roots
    return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

def train():
    model = QuadSolverNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("⏳ Generating data...")
    data, labels = generate_data(500_000)  # 🧪 Half a million samples!
    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.1)

    train_loader = torch.utils.data.DataLoader(
        list(zip(X_train, y_train)), batch_size=2048, shuffle=True)

    best_val_loss = float('inf')
    for epoch in range(300):
        start = time.time()
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            torch.save(model.state_dict(), "quad_solver_best.pth")

        if epoch % 10 == 0 or epoch == 299:
            print(f"🧠 Epoch {epoch} | Val Loss: {val_loss.item():.6f} | Time: {time.time()-start:.2f}s")

    print("✅ Model saved as quad_solver_best.pth")

# ✅ Use this to test any equation
def solve_custom(a, b, c):
    model = QuadSolverNN()
    model.load_state_dict(torch.load("quad_solver_best.pth"))
    model.eval()
    input_tensor = torch.tensor([[a / 100, b / 200, c / 500]], dtype=torch.float32)
    with torch.no_grad():
        pred = model(input_tensor).numpy()
    print(f"🧮 Predicted roots for {a}x² + ({b})x + {c} = 0: {pred}")

if __name__ == "__main__":
    train()
    # solve_custom(1, -3, 2)
