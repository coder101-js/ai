import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split

# --------------------------
# Bigger Brain Model
# --------------------------
class QuadSolverNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

# --------------------------
# Dataset Generator
# --------------------------
def generate_data(n):
    data = []
    labels = []
    while len(data) < n:
        a = np.random.uniform(1, 100)    # Wider range
        b = np.random.uniform(-200, 200)
        c = np.random.uniform(-500, 500)
        disc = b**2 - 4*a*c
        if disc < 0: continue  # Skip complex
        x1 = (-b + np.sqrt(disc)) / (2*a)
        x2 = (-b - np.sqrt(disc)) / (2*a)
        data.append([a / 100, b / 200, c / 500])  # Normalized input
        labels.append(sorted([x1, x2]))  # Order roots for consistency
    return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

# --------------------------
# Training Loop
# --------------------------
def train():
    model = QuadSolverNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Generate data
    data, labels = generate_data(100_000)
    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.1)

    best_val_loss = float('inf')

    for epoch in range(500):  # Train longer
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)

        if epoch % 25 == 0 or epoch == 499:
            print(f"Epoch {epoch} | Train Loss: {loss.item():.6f} | Val Loss: {val_loss.item():.6f}")

        # Save best model
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            torch.save(model.state_dict(), "quad_solver_best.pth")

    print("Training complete ✅")

    # Quick test
    test_in = torch.tensor([[1.0 / 100, -3.0 / 200, 2.0 / 500]])  # Normalized input
    pred = model(test_in).detach().numpy()
    print(f"Predicted roots (x² - 3x + 2): {pred}")

# --------------------------
# Load & Predict
# --------------------------
def solve_custom(a, b, c):
    model = QuadSolverNN()
    model.load_state_dict(torch.load("quad_solver_best.pth"))
    model.eval()
    inp = torch.tensor([[a / 100, b / 200, c / 500]], dtype=torch.float32)
    with torch.no_grad():
        pred = model(inp).numpy()[0]
    print(f"Predicted roots for {a}x² + {b}x + {c} = 0 → {pred[0]:.6f}, {pred[1]:.6f}")

# --------------------------
if __name__ == "__main__":
    train()
    # solve_custom(1, -3, 2)  # Uncomment to test with saved model
