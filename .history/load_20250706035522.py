import torch
import torch.nn as nn
import numpy as np

# 🧠 Must match the trained model's exact architecture!
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

# 🔁 Load the model
model = QuadSolverNN()
model.load_state_dict(torch.load("quad_solver_best.pth"))
model.eval()
print("✅ Model loaded and ready.")

# 🧪 Test input
a, b, c = 1, -3, 2
x = torch.tensor([[a, b, c]], dtype=torch.float32)

with torch.no_grad():
    result = model(x).numpy()[0]
    print(f"📣 Predicted roots: x1 = {result[0]:.4f}, x2 = {result[1]:.4f}")
