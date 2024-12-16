from abc import ABC, abstractmethod
from typing import TypeVar, Generic
from logging import Logger

from edgegen.evaluation import EvaluationEngine
from edgegen.search_space import ArchitectureGenerator
from edgegen.repository import ModelRepository

T = TypeVar('T')

class Pipeline(ABC):

    def __init__(self, archGenerator: ArchitectureGenerator[Generic[T]], 
                        evalEngine: EvaluationEngine[Generic[T]],
                        modelRepo: ModelRepository[Generic[T]],
                        logger: Logger) -> None:
        super().__init__()
        self.archGenerator = archGenerator
        self.evalEngine = evalEngine
        self.modelRepo = modelRepo
        self.logger = logger
        self._configure_logger()

    @abstractmethod
    def _configure_logger(self) -> None:
        pass
    
    @abstractmethod
    def run(self) -> None:
        pass