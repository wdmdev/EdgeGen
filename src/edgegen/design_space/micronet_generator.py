from edgegen.design_space import ArchitectureGenerator
from edgegen.design_space.architectures import MicroNet
from dataclasses import dataclass
from typing import List, Union
import ast

@dataclass
class MicroNetInputSpec:
    input_size:int
    num_classes:int
    teacher:bool
    ###### MicroNet Config ######
    block: str
    stem_ch: int
    stem_groups: Union[str, List[int]]
    stem_dilation: int
    stem_mode: str
    out_ch: int
    depth_sep: bool
    pointwise: str
    dropout_rate: float
    shuffle: bool
    ###### Activation Config ######
    activation_module: str
    activation_act_max: float
    activation_act_bias: bool
    activation_init_a_block3: List[float]
    activation_init_a: List[float]
    activation_init_b: List[float]
    activation_reduction: int
    act_ratio: int

    ########## Cfgs Config ##########
    cfgs_n: int
    stride: int 
    n_blocks: int 
    output_channel: int 
    kernel_size: int
    ch_exp1: int
    ch_exp2: int
    ch_per_group1: int
    ch_per_group2: int 
    groups_1x1_1: int 
    groups_1x1_2: int 
    groups_1x1_3: int 
    dy1: int
    dy2: int
    dy3: int


class MicroNetGenerator(ArchitectureGenerator):

    def get_input_spec(self) -> MicroNetInputSpec:
        return MicroNetInputSpec

    def generate(self, spec:MicroNetInputSpec) -> MicroNet:
        spec.stem_groups = ast.literal_eval(spec.stem_groups)
        return MicroNet(**spec.__dict__)