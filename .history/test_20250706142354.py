import torch
import torch.nn as nn
import numpy as np

# 🔧 Define your model architecture
class QuadraticSolverModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128),   # net.0
            nn.ReLU(),           # net.1
            nn.Dropout(0.1),     # net.2 (no weights)
            nn.Linear(128, 64),  # net.3 ✅
            nn.ReLU(),           # net.4
            nn.Dropout(0.1),     # net.5 (no weights)
            nn.Linear(64, 2)     # net.6 ✅
        )

    def forward(self, x):
        return self.net(x)



# 💾 Load your saved model
def load_model(path="quad_solver_best.pth"):
    model = QuadraticSolverModel()
    
    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state"])
    
    model.eval()
    return model


# 🧪 Predict roots using the model
def predict_roots(model, a, b, c):
    # Normalize using same scales as in training
    a_norm = a / 10.0
    b_norm = b / 20.0
    c_norm = c / 50.0

    input_tensor = torch.tensor([[a_norm, b_norm, c_norm]], dtype=torch.float32)
    with torch.no_grad():
        output = model(input_tensor)
    return output[0].tolist()


# 🧠 Compare with actual roots using math
def real_roots(a, b, c):
    return np.roots([a, b, c]).tolist()

# 🚀 Main test
if __name__ == "__main__":
    model = load_model()

    # 🎯 Example equation: 2x² + 5x - 3 = 0
    a, b, c = -1.0, , -3.0
    predicted = predict_roots(model, a, b, c)
    actual = real_roots(a, b, c)

    print(f"\n📐 Equation: {a}x² + {b}x + {c} = 0")
    print(f"🤖 Predicted roots: {predicted}")
    print(f"✅ Actual roots:    {actual}")
