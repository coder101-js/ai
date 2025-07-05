import torch
import torch.nn as nn
import numpy as np

# 👨‍🏫 Define the exact same architecture
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

# 🧠 Load model
model = QuadSolverNN()
model.load_state_dict(torch.load("quad_solver_best.pth"))
model.eval()
print("✅ Model loaded and ready.")
