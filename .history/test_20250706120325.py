import torch
import numpy as np
from ai import QuadSolverNN  # only if testing from another file

def solve_real(a, b, c):
    disc = b**2 - 4*a*c
    if disc < 0:
        return None
    x1 = (-b + np.sqrt(disc)) / (2*a)
    x2 = (-b - np.sqrt(disc)) / (2*a)
    return sorted([x1, x2])

def test_model(a, b, c, model_path="quad_solver_best.pth"):
    model = QuadSolverNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Normalize input just like during training
    input_tensor = torch.tensor([[a/10, b/20, c/50]], dtype=torch.float32)
    
    with torch.no_grad():
        pred = model(input_tensor).numpy().flatten()

    actual = solve_real(a, b, c)

    print(f"🧪 Equation: {a}x² + {b}x + {c} = 0")
    print(f"🤖 Model predicts: x1 = {pred[0]:.4f}, x2 = {pred[1]:.4f}")
    if actual:
        print(f"📐 Real roots:    x1 = {actual[0]:.4f}, x2 = {actual[1]:.4f}")
    else:
        print("⚠️ No real roots (discriminant < 0)")

# 🔍 Example test
test_model(a=3, b=-18, c=27)
