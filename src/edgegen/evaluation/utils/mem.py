import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch._subclasses.fake_tensor import FakeTensorMode
from edgegen import Bytes
from typing import Tuple

def estimate_torch_mem(architecture: nn.Module, input_size:Tuple[int], quant_size:int):
        class MemoryTrackingMode(TorchDispatchMode):
            def __init__(self, quant_size):
                super().__init__()
                self.current_memory = Bytes(size=0)
                self.peak_memory = Bytes(size=0)
                self.tensor_sizes = {}

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
                    mem = Bytes(tensor.numel() * quant_size)
                    self.current_memory += mem
                    self.peak_memory = max(self.peak_memory, self.current_memory)
                    self.tensor_sizes[id(tensor)] = mem

                # Track memory deallocations
                for arg in args:
                    if isinstance(arg, torch.Tensor) and id(arg) in self.tensor_sizes:
                        self.current_memory -= self.tensor_sizes.pop(id(arg))

                return outputs
        
        inputs = torch.randn(input_size)
        # Use FakeTensorMode and MemoryTrackingMode to estimate peak memory usage
        with FakeTensorMode(allow_non_fake_inputs=True) as fake_mode:
            fake_inputs = fake_mode.from_tensor(inputs)
            with MemoryTrackingMode(quant_size=quant_size) as mem_mode:
                # Simulate the forward pass
                fake_outputs = architecture(fake_inputs)
                # Get the peak memory usage
                peak_memory_usage = mem_mode.peak_memory
        
        return peak_memory_usage