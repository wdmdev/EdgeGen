from typing import Union
from .architecture_generator import ArchitectureGenerator
from .architectures import OFAProxylessNAS
from dataclasses import dataclass

#TODO - Fix OFA generator
@dataclass(repr=True)
class OFAProxylessNasInputSpec:
    """Class for input specification of OFAProxylessNAS architecture"""
    n_classes: int
    bn_momentum: float
    bn_eps: float
    dropout_rate: float
    base_stage_width: Union[None, int]
    width_mult_list: Union[float, list]
    ks_list: Union[int, list]
    expand_ratio_list: Union[int, list]
    depth_list: Union[int, list]
    no_mix_layer: bool



class OFAProxylessNasGenerator(ArchitectureGenerator):

    def get_input_spec(self) -> OFAProxylessNasInputSpec:
        return OFAProxylessNasInputSpec

    def generate(self, spec:OFAProxylessNasInputSpec) -> OFAProxylessNAS:

        return OFAProxylessNAS(**spec.__dict__)