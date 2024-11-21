import torch
import torch_scatter

# Check torch and CUDA availability
# Check torch and CUDA
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))

# Test torch-scatter's scatter_max on CUDA
try:
    x = torch.tensor([1, 2, 3], device="cuda")
    index = torch.tensor([0, 0, 1], device="cuda")
    max_values, argmax_indices = torch_scatter.scatter_max(x, index)
    print("scatter_max values:", max_values)
    print("scatter_max indices:", argmax_indices)
except Exception as err:
    print("Torch-scatter scatter_max error:", err)
