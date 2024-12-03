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
        self.result = ""

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
    
    def __str__(self) -> str:
        return "Constraint: " + self.name + "\nDescription: " + self.description + "\nResult: " + self.result