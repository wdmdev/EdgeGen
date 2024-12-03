from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class EvaluationResult():
        metrics: Dict[str, Any]
        satisfied_constraints: List[int]
        unsatisfied_constraints: List[int] 


        def __str__(self):
            return f"\nMetrics: {self.metrics}\nSatisfied Constraints: {self.satisfied_constraints}\nUnsatisfied Constraints: {self.unsatisfied_constraints}"