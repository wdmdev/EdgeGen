from dataclasses import dataclass
from .architecture_generator import ArchitectureGenerator
from .architectures import OFAProxylessNAS
from typing import Any, Dict, Union

@dataclass(repr=True)
class OFAProxylessNasInputSpec:
    """Class for input specification of OFAProxylessNAS architecture"""
    n_classes: int
    bn_param: tuple
    dropout_rate: float
    base_stage_width: Union[None, int]
    width_mult_list: Union[float, list]
    ks_list: Union[int, list]
    expand_ratio_list: Union[int, list]
    depth_list: Union[int, list]
    no_mix_layer: bool



class OFAProxylessNasGenerator(ArchitectureGenerator[OFAProxylessNAS]):

    def __init__(self) -> None:
        super().__init__()

    def generate(self, spec: Dict[str, Any]) -> OFAProxylessNAS:
        return OFAProxylessNAS(**spec)