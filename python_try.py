import torch
import torch_scatter

# Check torch and CUDA
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))

# Test torch-scatter
try:
    x = torch.tensor([1, 2, 3], device="cuda")
    y = torch_scatter.scatter_add(x, torch.tensor([0, 0, 1], device="cuda"))
    print("Torch-scatter works:", y)
except Exception as e:
    print("Torch-scatter error:", e)
