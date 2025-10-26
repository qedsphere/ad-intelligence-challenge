#!/usr/bin/env python3
"""
Test script to verify MPS (Metal Performance Shaders) GPU acceleration
"""

import torch
import sys

print("=" * 60)
print("Testing Apple GPU Acceleration (MPS)")
print("=" * 60)

# Check PyTorch version
print(f"\nPyTorch version: {torch.__version__}")

# Check if MPS is available
if torch.backends.mps.is_available():
    print(" MPS is available!")
    device = "mps"
else:
    print(" MPS is NOT available")
    print("   Will fall back to CPU (slower)")
    device = "cpu"

# Check if MPS is actually built
if torch.backends.mps.is_built():
    print("MPS is built into PyTorch")
else:
    print("  MPS is not built into this PyTorch installation")

# Test GPU with a simple operation
print(f"\n Testing computation on {device.upper()}...")

try:
    # Create a tensor and move it to device
    x = torch.randn(1000, 1000)
    x = x.to(device)
    
    # Perform computation
    y = torch.matmul(x, x)
    
    print(f" Successfully computed on {device.upper()}")
    print(f"   Tensor shape: {y.shape}")
    
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

# Summary
print("\n" + "=" * 60)
if device == "mps":
    print(" GPU Acceleration Ready!")
    print("   Models will run 2-3x faster")
else:
    print(" Using CPU only")
    print("   Processing will be slower but still work")
print("=" * 60)

