from abc import ABC, abstractmethod
from typing import TypeVar, Generic

T = TypeVar('T')
S = TypeVar('S')

class ArchitectureGenerator(ABC):
    @abstractmethod
    def generate(self, spec: Generic[S]) -> Generic[T]:
        """
        Generate architectures based on the given specification.

        Parameters
        ----------
        spec : dict
            The specification for the architecture generation.

        Returns
        -------
        Generic[T]
            The generated architecture.
        """
        pass