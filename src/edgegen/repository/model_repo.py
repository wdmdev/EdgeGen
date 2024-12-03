from abc import ABC, abstractmethod
from typing import TypeVar, Generic
from pathlib import Path

T = TypeVar('T')

class ModelRepository(ABC):
    """
    Base class for model repository
    """

    @abstractmethod
    def save(self, model: Generic[T], path:Path) -> None:
        pass
    
    @abstractmethod
    def load(self, path:Path) -> Generic[T]:
        pass