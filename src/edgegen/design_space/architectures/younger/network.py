# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2023-09-14 12:22
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.

#### DISCKAIMER####
# The original code has been modified by reducing it to the parts essential for this project.

import os
import pathlib
import networkx
from edgegen.design_space.architectures.younger.utils.io import load_json, load_pickle, save_pickle
from typing import Generator


class Network(object):
    def __init__(
            self,
            data_path: str | pathlib.Path | None = None,
            graph: networkx.DiGraph | None = None,
    ) -> None:
        self._graph_filename = 'graph.pkl'
        self._info_filename = 'info.json'
        self.data_path = data_path

        if graph:
            graph = graph or networkx.DiGraph()
            # node_attributes:
            #   1. type='operator':
            #     name
            #     doc_string
            #     operator
            #     operands
            #     results
            #     attributes
            #   2. type='input':
            #     graph_inputs
            #   3. type='output':
            #     graph_outputs
            #   4. type='constant':
            #     graph_constants

            # edge_attributes:
            #   head_index
            #   tail_index
            #   emit_index
            #   trap_index
            #   dataflow
            #   default_value


            self._graph = graph

            ir_version: int = graph.graph.get('ir_version', None)
            opset_import: dict[str, int] = graph.graph.get('opset_import', None)
            producer_name: str | None = graph.graph.get('producer_name', None)
            producer_version: str | None = graph.graph.get('producer_version', None)
            domain: str | None = graph.graph.get('domain', None)
            model_version: int | None = graph.graph.get('model_version', None)
            doc_string: str | None = graph.graph.get('doc_string', None)
            metadata_props: list[dict[str, str]] | None = graph.graph.get('metadata_props', None)

            self._info = dict(
                ir_version = ir_version,
                opset_import = opset_import,
                producer_name = producer_name,
                producer_version = producer_version,
                domain = domain,
                model_version = model_version,
                doc_string = doc_string,
                metadata_props = metadata_props,
            )
        elif data_path:
            if isinstance(data_path, str):
                data_path = pathlib.Path(data_path)
            assert not data_path.is_file(), f'The provided data path \"{data_path.absolute()}\" must be changed to a network folder.'
            self.load(data_path)

    @property
    def graph(self) -> networkx.DiGraph:
        return self._graph

    @property
    def info(self) -> dict:
        return self._info

    def load(self, network_dirpath: pathlib.Path) -> None:
        assert network_dirpath.is_dir(), f'There is no \"Network\" can be loaded from the specified directory \"{network_dirpath.absolute()}\".'
        info_filepath = network_dirpath.joinpath(self._info_filename)
        self._load_info(info_filepath)
        graph_filepath = network_dirpath.joinpath(self._graph_filename)
        self._load_graph(graph_filepath)
        return 

    def _load_graph(self, graph_filepath: pathlib.Path) -> None:
        assert graph_filepath.is_file(), f'There is no \"graph\" can be loaded from the specified path \"{graph_filepath.absolute()}\".'
        self._graph = load_pickle(graph_filepath)
        return

    def _load_info(self, info_filepath: pathlib.Path) -> None:
        assert info_filepath.is_file(), f'There is no \"INFO\" can be loaded from the specified path \"{info_filepath.absolute()}\".'
        self._info = load_json(info_filepath)
        return

    def save_graph(self) -> None:
        if not self.data_path:
            raise ValueError('The data path must be specified before saving the network graph.')
        save_pickle(self._graph, self.data_path)
        return


def count_networks(data_path: pathlib.Path | str) -> int:
    count = 0

    for _, dirs, _ in os.walk(data_path):
        if 'network' in dirs:
            count += 1

    return count

def load_networks(data_path: pathlib.Path | str) -> Generator[Network, None, None]:
    for root, dirs, files in os.walk(data_path):
        if 'network' in dirs:
            network_path = os.path.join(root, 'network')
            NN = Network(data_path=network_path)
            NN.load(pathlib.Path(network_path))
            yield NN