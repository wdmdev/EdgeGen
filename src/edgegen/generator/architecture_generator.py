import random
import torch
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Union

T = TypeVar('T')
S = TypeVar('S')

class ArchitectureGenerator(ABC):

    def __init__(self, seed: Union[int, None]=None) -> None:
        super().__init__()
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

    @abstractmethod
    def get_input_spec(self) -> Generic[S]:
        ...


    @abstractmethod
    def generate(self, spec:Generic[S]) -> Generic[T]:
        """
        Generate architectures based on the given specification.

        Parameters
        ----------
        spec : Generic[S]
            The specification for the architecture generation.

        Returns
        -------
        Generic[T]
            The generated architecture.
        """
        pass