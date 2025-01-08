from edgegen.design_space import ArchitectureGenerator
from edgegen.design_space.architectures import YoungerNet
from dataclasses import dataclass
from typing import List

# The input spec is kept for consistency with the ArchitectureGenerator interface
@dataclass
class YoungerInputSpec:
    valid_ops: List[str]
    valid_input_ops: List[str]
    valid_output_ops: List[str]


class YoungerGenerator(ArchitectureGenerator):
    def __init__(self) -> None:
        super().__init__()
    
    def get_input_spec(self) -> YoungerInputSpec:
        return YoungerInputSpec
    
    def generate(self, spec:YoungerInputSpec) -> YoungerNet:
        return YoungerNet(**spec.__dict__)