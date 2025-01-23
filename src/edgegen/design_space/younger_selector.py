from edgegen.design_space import ArchitectureGenerator
from dataclasses import dataclass
from collections.abc import Generator
from typing import List, Union
import networkx as nx
import os
from pathlib import Path
from edgegen.design_space.architectures.younger.network import Network
from edgegen.design_space.architectures.younger.utils.hashing import make_json_serializable, hash_node
import json
from tqdm import tqdm

# The input spec is kept for consistency with the ArchitectureGenerator interface
@dataclass
class YoungerSelectorInputSpec:
    valid_ops: Union[List[str], None]
    valid_input_ops: Union[List[str], None]
    valid_output_ops: Union[List[str], None]
    hash_nodes: bool


class YoungerSelector(ArchitectureGenerator):
    def __init__(self) -> None:
        super().__init__()

    def get_input_spec(self) -> YoungerSelectorInputSpec:
        return YoungerSelectorInputSpec

    def generate(self, spec:YoungerSelectorInputSpec) -> Generator[nx.DiGraph, None, None]:
        network_folder = Path(__file__).parent / 'architectures'/ 'younger' / 'networks'
        config_path = Path(__file__).parent / 'architectures' / 'younger' / 'config'
        valid_ops = spec.valid_ops
        valid_input_ops = spec.valid_input_ops
        valid_output_ops = spec.valid_output_ops

        if valid_ops is None:
            with open(os.path.join(config_path, "valid_ops.json")) as f:
                valid_ops = json.load(f)['operators']

        if valid_input_ops is None:
            with open(os.path.join(config_path, "valid_input_ops.json")) as f:
                valid_input_ops = json.load(f)['operators'] 

        if valid_output_ops is None:
            with open(os.path.join(config_path, "valid_output_ops.json")) as f:
                valid_output_ops = json.load(f)['operators']

        print(f"Building architecture collection from {network_folder}")

        for graph_path in tqdm(network_folder.glob('**/network'), total=sum(1 for _ in network_folder.glob('**/network'))):
            network = Network(graph_path)
            graph = network.graph

            # Check if the graph contains any invalid ops
            if any([n for n, attr in graph.nodes(data=True) if attr['operator']['op_type'] not in valid_input_ops]):
                continue

            if spec.hash_nodes:
                node_hashes = []
                for node in graph.nodes(data=True):
                    graph.nodes[node[0]].update(make_json_serializable(node[1]))
                    node_hashes.append(hash_node(node[1]))

                nx.relabel_nodes(graph, dict(zip(graph.nodes, node_hashes)), copy=False)
            
            yield graph
        