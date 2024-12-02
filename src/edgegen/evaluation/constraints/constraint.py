from abc import ABC, abstractmethod
from typing import TypeVar, Generic

T = TypeVar('T')

class Constraint(ABC):
    """
    Base class for constraints.
    """

    def __init__(self, name:str, description:str) -> None:
        super().__init__()
        self.name = name
        self.description = description

    @abstractmethod
    def is_satisfied(self, architecture: Generic[T]) -> bool:
        """
        Check if the given architecture satisfies the constraint.

        Parameters
        ----------
        architecture : T
            The architecture to be checked.

        Returns
        -------
        bool
            True if the architecture satisfies the constraint, False otherwise.
        """
        ...