# Quick test to verify the fix
import torch
from src.models import EnhancedLSTMModel

# Create test model
model = EnhancedLSTMModel(input_size=10, hidden_size=64)

# Create test input (batch_size=2, seq_len=5, features=10)
test_input = torch.randn(2, 5, 10)

# Forward pass
output = model(test_input)

print(f"✅ Model forward pass successful!")
print(f"   Input shape: {test_input.shape}")
print(f"   Output shape: {output.shape}")
print(f"   Output range: {output.min().item():.4f} to {output.max().item():.4f}")

# Check if output is bounded (indication of activation)
if output.min() >= -2 and output.max() <= 2:
    print("✅ Output appears bounded - activation function working!")
    print("🎉 FIX LIKELY SUCCESSFUL!")
else:
    print("⚠️ Output range seems unbounded - activation may not be applied")