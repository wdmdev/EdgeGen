from torch import nn
from edgegen.evaluation import Constraint
from edgegen import Bytes
from edgegen.evaluation.utils import estimate_torch_flash


class PyTorchFlashConstraint(Constraint):
    """
    Constraint to check if the flash storage usage of a PyTorch model is within the specified limits.
    """

    def __init__(self, name:str, description:str, 
                 max_flash_limit: Bytes, 
                 quant_size: int = 4) -> None:
        super().__init__(name, description)
        self.max_flash_limit = max_flash_limit
        self.flash_storage_usage = Bytes(size=0)
        self.quant_size = quant_size

    def is_satisfied(self, architecture: nn.Module) -> bool:
        """
        Check if the flash storage usage of the given PyTorch model is within the specified limits.

        Parameters
        ----------
        architecture : nn.Module
            The PyTorch model to be checked.

        Returns
        -------
        bool
            True if the flash storage usage of the PyTorch model is within the specified limits, False otherwise.
        """
        self.flash_storage_usage = estimate_torch_flash(architecture, self.quant_size)
        is_valid = self.flash_storage_usage <= self.max_flash_limit
        if is_valid:
            self.result = "Model Flash: " + str(self.flash_storage_usage) + " <= " + str(self.max_flash_limit)
        else:
            self.result = "Model Flash: " + str(self.flash_storage_usage) + " > " + str(self.max_flash_limit)
        
        return is_valid



if __name__ == '__main__':
    # Define the same AdaptiveCNN model as before
    from torch.functional import F

    class AdaptiveCNN(nn.Module):
        def __init__(self, input_channels, num_classes):
            super(AdaptiveCNN, self).__init__()
            self.conv1 = nn.Conv2d(input_channels, 60, kernel_size=3)  # First convolutional layer
            self.conv2 = nn.Conv2d(60, 600, kernel_size=3)              # Second convolutional layer
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
    input_size = (3, 128, 128)

    big_model = AdaptiveCNN(input_channels=input_size[0], num_classes=num_classes)

    # Create a PyTorchFlashConstraint object
    max_flash_limit = Bytes.from_MB(0.9)
    flash_constraint = PyTorchFlashConstraint(max_flash_limit=max_flash_limit, quant_size=4)

    # Check if the flash storage usage of the model is within the specified limits
    print("\nChecking flash storage usage of the model...")
    print(f"Max flash storage limit: {max_flash_limit.to_KB()} KB")

    is_satisfied = flash_constraint.is_satisfied(big_model)
    print(f"Flash storage usage within limits: {is_satisfied}")
    print(f"Flash storage usage: {flash_constraint.flash_storage_usage.to_MB():.2f} MB")

    print("\nChecking flash storage usage of the model if it is quantized to int8...")
    flash_constraint.quant_size = 1
    is_satisfied = flash_constraint.is_satisfied(big_model)
    print(f"Flash storage usage within limits: {is_satisfied}")
    print(f"Flash storage usage: {flash_constraint.flash_storage_usage.to_MB():.2f} MB")
