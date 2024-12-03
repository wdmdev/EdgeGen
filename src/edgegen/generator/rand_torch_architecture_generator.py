import random
from edgegen.generator import ArchitectureGenerator
from torch import nn
from typing import Tuple, List
from dataclasses import dataclass

@dataclass
class RandomPytorchArchitectureSpec:
    min_depth: int
    max_depth: int
    min_filters: int
    max_filters: int
    min_neurons: int
    max_neurons: int
    kernel_sizes: List[int]
    strides: List[int]
    min_fc_layers: int
    max_fc_layers: int
    num_classes: int
    input_channels: int
    input_size: Tuple[int, int, int]

#TODO - Bug fix
class RandomPytorchArchitectureGenerator(ArchitectureGenerator):

    def get_input_spec(self) -> RandomPytorchArchitectureSpec:
        return RandomPytorchArchitectureSpec

    def generate(self, spec: RandomPytorchArchitectureSpec) -> nn.Module:

        layers = []
        in_channels = self.spec.input_channels
        input_height, input_width = self.spec.input_size[1], self.spec.input_size[2]

        # Randomly determine the number of convolutional layers
        num_conv_layers = random.randint(self.spec.min_depth, self.spec.max_depth)

        for _ in range(num_conv_layers):
            out_channels = random.randint(self.spec.min_filters, self.spec.max_filters)
            kernel_size = random.choice(self.spec.kernel_sizes)
            stride = random.choice(self.spec.strides)
            padding = (kernel_size - 1) // 2  # To maintain the same output size

            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # Downsample by a factor of 2

            # Update input dimensions for the next layer
            input_height = (input_height + 2 * padding - kernel_size) // stride + 1
            input_width = (input_width + 2 * padding - kernel_size) // stride + 1
            input_height = input_height // 2  # MaxPool2d downsampling
            input_width = input_width // 2

            in_channels = out_channels

        # Flatten the output from the convolutional layers
        layers.append(nn.Flatten())

        # Calculate the flattened feature size
        flattened_size = in_channels * input_height * input_width

        # Randomly determine the number of fully connected layers
        num_fc_layers = random.randint(self.spec.min_fc_layers, self.spec.max_fc_layers)
        in_features = flattened_size

        for _ in range(num_fc_layers):
            out_features = random.randint(self.spec.min_neurons, self.spec.max_neurons)
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            in_features = out_features

        # Final output layer
        layers.append(nn.Linear(in_features, self.spec.num_classes))

        # Create the sequential model
        model = nn.Sequential(*layers)
        return model
