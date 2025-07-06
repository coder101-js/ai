import torch
import torch.nn as nn
import numpy as np

# Define the model architecture exactly as in training
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

def load_model(checkpoint_path="quad_solver_best.pth"):
    model = QuadSolverNN()
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    print(f"✅ Model loaded from {checkpoint_path}")
    return model

def predict_roots(model, a, b, c):
    # Normalize input same way as training
    x = torch.tensor([[a/10, b/20, c/50]], dtype=torch.float32)
    with torch.no_grad():
        roots = model(x).numpy().flatten()
    return roots

if __name__ == "__main__":
    model = load_model()

    # Example: Predict roots for equation 2x² + 3x + 1 = 0
    a, b, c = 2, 3, 1
    roots = predict_roots(model, a, b, c)
    print(f"Quadratic roots prediction for {a}x² + {b}x + {c} = 0 → {roots}")
