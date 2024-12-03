from torch import nn
from edgegen import Bytes

def estimate_torch_flash(architecture: nn.Module, quant_size) -> Bytes:
    total_storage = 0
    for param in architecture.parameters():
        # Calculate storage for each parameter
        total_storage += param.numel() * quant_size
    flash_storage_usage = Bytes(size=total_storage)

    return flash_storage_usage