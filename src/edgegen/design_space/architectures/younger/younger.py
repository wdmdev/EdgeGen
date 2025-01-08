import os
import json
import pickle as pkl
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import networkx as nx
import random
import uuid

from edgegen.design_space.architectures.younger.network import Network
from edgegen.design_space.architectures.younger.utils.hashing import hash_node

class YoungerNet:

    def __init__(self, valid_ops=None, valid_input_ops=None, valid_output_ops=None):
        self.data_path = Path(__file__).parent / 'networks'
        self.config_path = Path(__file__).parent / 'config'
        self.graph_path = Path(__file__).parent / 'super_graph.pkl'
        self.valid_ops = valid_ops
        self.valid_input_ops = valid_input_ops
        self.valid_output_ops = valid_output_ops

        if valid_ops is None:
            with open(os.path.join(self.config_path, "valid_ops.json")) as f:
                self.valid_ops = json.load(f)['operators']

        if valid_input_ops is None:
            with open(os.path.join(self.config_path, "valid_input_ops.json")) as f:
                self.valid_input_ops = json.load(f)['operators'] 

        if valid_output_ops is None:
            with open(os.path.join(self.config_path, "valid_output_ops.json")) as f:
                self.valid_output_ops = json.load(f)['operators']

        if os.path.exists(self.graph_path):
            print(f"Loading super_graph data from {self.graph_path}")
            with open(self.graph_path) as f:
                graph_data = pkl.load(f)
                self.input_nodes = graph_data['input_nodes']
                self.output_nodes = graph_data['output_nodes']
                self.super_graph = graph_data['super_graph']
        else:
            # find max_workers
            max_workers = os.cpu_count() - 1
            if max_workers < 1:
                max_workers = 1
            print("Creating super graph from the Younger dataset...")
            print(f"Using {max_workers} workers to process graphs.")
            graph_data = self.__iterate_graphs(self.data_path, max_workers)
            self.input_nodes = graph_data['input_nodes']
            self.output_nodes = graph_data['output_nodes']
            self.super_graph = graph_data['super_graph']

            print(f"Saving super_graph data to {self.graph_path}")
            with open(self.graph_path, 'wb') as f:
                pkl.dump(graph_data, f)
        
        self.__clean_super_graph()


    
    def __process_single_graph(self, graph_path):
        """Process a single graph to extract input/output nodes, all unique nodes, and edges."""
        NN = Network()
        NN.load(Path(graph_path))
        graph = NN.graph

        # Find input nodes (nodes with no incoming edges) and output nodes (nodes with no outgoing edges)
        input_nodes = {hash_node(node) for name, node in graph.nodes(data=True) if graph.in_degree(name) == 0}
        output_nodes = {hash_node(node) for name, node in graph.nodes(data=True) if graph.out_degree(name) == 0}

        # Extract all nodes and hash them to create unique identifiers
        all_nodes = {}
        for _, node in graph.nodes(data=True):
            try:
                node_hash = hash_node(node)
                all_nodes[node_hash] = node
            except Exception as e:
                print(f"Error processing node: {node}: {e}")

        # Extract all edges as (source, target) pairs with hashed nodes
        edges = [(hash_node(graph.nodes[source]), hash_node(graph.nodes[target])) for source, target in graph.edges()]

        return {
            'input_nodes': input_nodes,
            'output_nodes': output_nodes,
            'all_nodes': all_nodes,
            'edges': edges
        }

    def __iterate_graphs(self, graph_paths, max_workers=4):
        """Run multiple processes in parallel to analyze graphs and combine the results."""

        all_input_nodes = set()
        all_output_nodes = set()
        all_unique_nodes = dict()
        edge_counts = defaultdict(int)  # Edge weights (counts) for the combined graph

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {executor.submit(self.__process_single_graph, path): path for path in graph_paths}

            for future in tqdm(as_completed(future_to_path), total=len(future_to_path), desc="Processing graphs"):
                try:
                    result = future.result()

                    # Collect all distinct input and output nodes
                    all_input_nodes.update(result['input_nodes'])
                    all_output_nodes.update(result['output_nodes'])
                    all_unique_nodes.update(result['all_nodes'])

                    # Count how often each edge appears across all graphs
                    for edge in result['edges']:
                        edge_counts[edge] += 1

                except Exception as e:
                    print(f"Error processing graph: {future_to_path[future]}: {e}")

        # Create the combined graph with weighted edges
        super_graph = nx.DiGraph()
        for (source, target), weight in edge_counts.items():
            super_graph.add_edge(source, target, weight=weight)

        for node_hash, node in all_unique_nodes.items():
            if node_hash in super_graph.nodes:
                super_graph.nodes[node_hash].update(node)
            else:
                super_graph.add_node(node_hash, **node)

        return {
            'input_nodes': all_input_nodes,
            'output_nodes': all_output_nodes,
            'super_graph': super_graph,
        }
    
    def __clean_super_graph(self) -> None:
        """
        Cleans the super graph by removing invalid nodes and isolated nodes.
    
        Args:
            super_graph (nx.Graph): The input super graph.
            input_nodes (list): List of input nodes.
            output_nodes (list): List of output nodes.
            valid_ops (list): List of valid operator types.
    
        Returns:
            tuple[nx.Graph, list, list]: The cleaned graph, updated input nodes, and updated output nodes.
        """
        get_node_op_type = lambda graph, node_id: graph.nodes[node_id]['operator'].get('op_type', 'unknown')

        # Step 1: Remove all invalid nodes and their corresponding edges
        nodes_to_remove = [
            node for node in self.super_graph.nodes 
            if get_node_op_type(self.super_graph, node) not in self.valid_ops
        ]
    
        # Remove invalid nodes from the graph
        self.super_graph.remove_nodes_from(nodes_to_remove)
    
        # Step 2: Remove all nodes that are isolated (i.e., have no edges)
        isolated_nodes = list(nx.isolates(self.super_graph))
        self.super_graph.remove_nodes_from(isolated_nodes)

        # Remove isolated nodes from input and output nodes
        input_nodes = [node for node in input_nodes if node not in isolated_nodes]
        output_nodes = [node for node in output_nodes if node not in isolated_nodes]

        # Step 3: Find all input and output nodes that are valid
        self.input_nodes = [node for node in input_nodes if (node in self.super_graph.nodes and get_node_op_type(self.super_graph, node) in self.valid_input_ops)]
        self.output_nodes = [node for node in output_nodes if (node in self.super_graph.nodes and get_node_op_type(self.super_graph, node) in self.valid_output_ops)]

        # Step 4: Remove all roots and leafs that are not input or output nodes
        roots_leafs_to_remove = [node for node in self.super_graph.nodes if 
                                 (self.super_graph.out_degree(node) == 0 and node not in input_nodes) or
                                 (self.super_graph.in_degree(node) == 0 and node not in output_nodes)]
        self.super_graph = self.super_graph.remove_nodes_from(roots_leafs_to_remove)