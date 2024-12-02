from abc import ABC, abstractmethod
from typing import TypeVar, List, Generic
from .constraint import Constraint

T = TypeVar('T')

class ConstraintManager():
    """
    Base class for constraint managers.
    """

    def __init__(self, constraints:List[Constraint]) -> None:
        super().__init__()
        self.constraints = constraints

    @abstractmethod
    def validate(self, architecture: Generic[T]) -> List[bool]:
        """
        Validate the given architecture based on the constraints.

        Parameters
        ----------
        architecture : Generic[T]
            The architecture to validate.

        Returns
        -------
        List[bool]
            A list of boolean values indicating whether the architecture satisfies the constraints
        """
        return [c.is_satisfied(architecture) for c in self.constraints]