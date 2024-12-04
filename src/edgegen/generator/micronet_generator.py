from edgegen.generator import ArchitectureGenerator
from edgegen.generator.architectures import MicroNet
from dataclasses import dataclass
from typing import List

@dataclass
class MicroNetActivationSpec:
    module: str
    act_max: float
    act_bias: bool
    init_a_block3: List[float]
    init_a: List[float]
    init_b: List[float]
    reduction: int

@dataclass
class MicroNetInputSpec:
    input_size:int
    num_classes:int
    teacher:bool
    ###### MicroNet Config ######
    block: str
    stem_ch: int
    stem_groups: List[int]
    stem_dilation: int
    stem_mode: str
    out_ch: int
    depth_sep: bool
    pointwise: str
    dropout_rate: float
    shuffle: bool
    ###### Activation Config ######
    activation_cfg: MicroNetActivationSpec




class MicroNetGenerator(ArchitectureGenerator):

    def get_input_spec(self) -> MicroNetInputSpec:
        return MicroNetInputSpec

    def generate(self, spec:MicroNetInputSpec) -> MicroNet:

        return MicroNet(**spec.__dict__)