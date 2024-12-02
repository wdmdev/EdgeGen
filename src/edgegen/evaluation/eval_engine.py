from edgegen.evaluation import ConstraintManager, Constraint, EvaluationResult
from typing import TypeVar, Generic, List, Callable, Dict, Any

T = TypeVar('T')

#TODO - Produce a better evaluation report that can be elegantly logged
class EvaluationEngine():
    def __init__(self, constraint_manager: ConstraintManager,
                 metrics: Dict[str, Callable[[Generic[T]], Any]]) -> None:
        self.constraint_manager = constraint_manager
        self.metrics = metrics

    def evaluate(self, architecture: Generic[T]) -> EvaluationResult:
        constraint_results = self.constraint_manager.validate(architecture)
        metric_results = {name: metric(architecture) for name, metric in self.metrics.items()}

        return EvaluationResult(metrics=metric_results,
                                satisfied_constraints=[i for i, is_satisfied in enumerate(constraint_results) if is_satisfied],
                                unsatisfied_constraints=[i for i, is_satisfied in enumerate(constraint_results) if not is_satisfied])

    def get_satisfied_constraints(self, architecture: Generic[T]) -> List[Constraint]:
        return [c for c in self.constraint_manager.constraints if c.is_satisfied(architecture)]
    
    def get_unsatisfied_constraints(self, architecture: Generic[T]) -> List[Constraint]:
        return [c for c in self.constraint_manager.constraints if not c.is_satisfied(architecture)]


if __name__ == "__main__":
    from torch import nn
    from torch.functional import F
    from edgegen import Bytes
    from edgegen.evaluation import ConstraintManager, PyTorchMemoryConstraint, PyTorchFlashConstraint
    # Create test constraint manager and constraints
    input_size = (3, 128, 128)
    constraints = [
        PyTorchMemoryConstraint(input_size, max_memory_limit=Bytes.from_KB(600), quant_size=4),
        PyTorchFlashConstraint(max_flash_limit=Bytes.from_KB(900), quant_size=4),
    ]
    constraint_manager = ConstraintManager(constraints=constraints)

    # Create test evaluation engine
    metrics: Dict[str, Callable[[Generic[T]], Any]] = {
        "model_param_type": lambda arch: set(p.dtype for p in arch.parameters()),
    }

    eval_engine = EvaluationEngine(constraint_manager=constraint_manager, metrics=metrics)

    # Test evaluation
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

    model = AdaptiveCNN(input_channels=input_size[0], num_classes=num_classes)

    eval_result = eval_engine.evaluate(model)
    print(eval_result)

