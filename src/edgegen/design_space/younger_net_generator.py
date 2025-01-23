from edgegen.design_space import ArchitectureGenerator
from edgegen.design_space.architectures import YoungerNet
from dataclasses import dataclass
from typing import List, Union

@dataclass
class YoungerNetInputSpec:
    valid_ops: Union[List[str], None]
    valid_input_ops: Union[List[str], None]
    valid_output_ops: Union[List[str], None]


class YoungerNetGenerator(ArchitectureGenerator):
    def __init__(self) -> None:
        super().__init__()
    
    def get_input_spec(self) -> YoungerNetInputSpec:
        return YoungerNetInputSpec
    
    def generate(self, spec:YoungerNetInputSpec) -> YoungerNet:
        return YoungerNet(**spec.__dict__)