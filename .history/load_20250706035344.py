import torch
import torch.nn as nn
import numpy as np

# ✅ Match the original training architecture (NO DROPOUT)
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

# 🧠 Load model
model = QuadSolverNN()
model.load_state_dict(torch.load("quad_solver_best.pth"))
model.eval()
print("✅ Model loaded and ready.")
