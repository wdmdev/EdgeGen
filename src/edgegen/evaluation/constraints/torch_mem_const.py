import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch._subclasses.fake_tensor import FakeTensorMode
from edgegen.evaluation import Constraint
from edgegen import Bytes
from typing import Tuple


class PyTorchMemoryConstraint(Constraint):
    """
    Constraint to check if the memory usage of a PyTorch model is within the specified limits.
    """

    def __init__(self, input_size: Tuple[int], max_memory_limit: Bytes,
                 quant_size: int=1) -> None:
        super().__init__("PyTorch Memory Constraint",
                         "Check if the memory usage of a PyTorch model is within the specified limits.")
        self.input_size = input_size
        self.max_memory_limit = max_memory_limit
        self.peak_memory_usage = Bytes(size=0)
        self.quant_size = quant_size

    def is_satisfied(self, architecture: nn.Module) -> bool:
        """
        Check if the memory usage of the given PyTorch model is within the specified limits.

        Parameters
        ----------
        architecture : PyTorchArchitecture
            The PyTorch model to be checked.

        Returns
        -------
        bool
            True if the memory usage of the PyTorch model is within the specified limits, False otherwise.
        """
        class MemoryTrackingMode(TorchDispatchMode):
            def __init__(self, quant_size):
                super().__init__()
                self.current_memory = Bytes(size=0)
                self.peak_memory = Bytes(size=0)
                self.tensor_sizes = {}
                self.quant_size = quant_size

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # Call the function to get outputs
                outputs = func(*args, **(kwargs or {}))

                # Track memory allocations
                tensors = []
                if isinstance(outputs, torch.Tensor):
                    tensors = [outputs]
                elif isinstance(outputs, (tuple, list)):
                    tensors = [out for out in outputs if isinstance(out, torch.Tensor)]

                for tensor in tensors:
                    mem = Bytes(tensor.numel() * self.quant_size)
                    self.current_memory += mem
                    self.peak_memory = max(self.peak_memory, self.current_memory)
                    self.tensor_sizes[id(tensor)] = mem

                # Track memory deallocations
                for arg in args:
                    if isinstance(arg, torch.Tensor) and id(arg) in self.tensor_sizes:
                        self.current_memory -= self.tensor_sizes.pop(id(arg))

                return outputs
        
        inputs = torch.randn(self.input_size)
        # Use FakeTensorMode and MemoryTrackingMode to estimate peak memory usage
        with FakeTensorMode(allow_non_fake_inputs=True) as fake_mode:
            fake_inputs = fake_mode.from_tensor(inputs)
            with MemoryTrackingMode(quant_size=self.quant_size) as mem_mode:
                # Simulate the forward pass
                fake_outputs = architecture(fake_inputs)
                # Get the peak memory usage
                self.peak_memory_usage = mem_mode.peak_memory

        return self.peak_memory_usage.size <= self.max_memory_limit.size


if __name__ == '__main__':
    from torch.functional import F

    class AdaptiveCNN(nn.Module):
        def __init__(self, input_channels, num_classes):
            super(AdaptiveCNN, self).__init__()
            self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=3)  # First convolutional layer
            self.conv2 = nn.Conv2d(6, 16, kernel_size=3)              # Second convolutional layer
            self.fc1 = None  # Placeholder for the first fully connected layer
            self.num_classes = num_classes

        def forward(self, x):
            x = F.relu(self.conv1(x))        # Apply first convolution and ReLU activation
            x = F.max_pool2d(x, kernel_size=2)  # Apply 2x2 max pooling
            x = F.relu(self.conv2(x))        # Apply second convolution and ReLU activation
            x = F.max_pool2d(x, kernel_size=2)  # Apply 2x2 max pooling

            if self.fc1 is None:
                # Dynamically compute the size of the flattened features
                num_features = x.numel() // x.size(0)
                self.fc1 = nn.Linear(num_features, self.num_classes).to(x.device)  # Initialize fully connected layer

            x = x.view(x.size(0), -1)        # Flatten the tensor
            x = self.fc1(x)                  # Apply the fully connected
            return x

    num_classes = 10
    input_size = (3,128,128)

    big_model = AdaptiveCNN(input_channels=input_size[0], num_classes=num_classes)

    # Create a PyTorchMemoryConstraint object
    max_memory_limit = Bytes.from_KB(550)
    memory_constraint = PyTorchMemoryConstraint(input_size, max_memory_limit, quant_size=4)

    # Check if the memory usage of the model is within the specified limits
    print("Checking memory usage of the models...")
    print(f"Input size: {input_size}")
    print(f"Max memory limit: {max_memory_limit.to_KB()} KB")

    print("\nChecking memory usage of the model that exceeds the memory limit...")
    print("Big model:")
    print(f"Model dtypes: {set(p.dtype for p in big_model.parameters())}")
    is_satisfied = memory_constraint.is_satisfied(big_model)
    print(f"Memory usage within limits: {is_satisfied}")
    print(f"Peak memory usage: {memory_constraint.peak_memory_usage.to_KB():.2f} KB")

    print("\nChecking memory usage of the model if it is quantized to int8...")
    print("Quantized int8 model:")
    memory_constraint.quant_size = 1
    is_satisfied = memory_constraint.is_satisfied(big_model)
    print(f"Memory usage within limits: {is_satisfied}")
    print(f"Peak memory usage: {memory_constraint.peak_memory_usage.to_KB():.2f} KB")
