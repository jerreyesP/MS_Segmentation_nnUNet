import torch

# Now we should check and follow all the steps for initiate
print(f"Check the version of cudnn: {torch.backends.cudnn.version()}")

# Check the version of torch like
print(f"Check the version of pytorch: {torch.__version__}")

# Create a tensor on the GPU.
x = torch.cuda.FloatTensor(5, 3)
print(f"tensor on the GPU:\n{x}")

# Get the amount of memory allocated on the GPU.
memory_allocated = torch.cuda.memory_allocated()

# Print the amount of memory allocated on the GPU.
print(f"memory_allocated: {memory_allocated}")

# Cheacking GPU avaibility for pytorch
torch.cuda.is_available()