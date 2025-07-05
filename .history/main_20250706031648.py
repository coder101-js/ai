import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Simple feedforward model
class QuadSolverNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32),  # input: a,b,c
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2)   # output: x1, x2
        )
        
    def forward(self, x):
        return self.net(x)

# Generate synthetic data
def generate_data(n):
    data = []
    labels = []
    for _ in range(n):
        a = np.random.uniform(1, 10)
        b = np.random.uniform(-20, 20)
        c = np.random.uniform(-50, 50)
        disc = b**2 - 4*a*c
        if disc < 0:
            continue  # skip complex roots for now
        x1 = (-b + np.sqrt(disc)) / (2*a)
        x2 = (-b - np.sqrt(disc)) / (2*a)
        data.append([a, b, c])
        labels.append([x1, x2])
    return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

# Training loop
def train():
    model = QuadSolverNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    data, labels = generate_data(10000)

    for epoch in range(200):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"Epoch {epoch} Loss: {loss.item():.6f}")

    # Test sample
    test_in = torch.tensor([[1.0, -3.0, 2.0]])  # eq: x^2 - 3x + 2 = 0, roots 2 and 1
    pred = model(test_in).detach().numpy()
    print(f"Predicted roots: {pred}")

if __name__ == "__main__":
    train()
