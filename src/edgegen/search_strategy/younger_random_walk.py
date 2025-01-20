from edgegen.evaluation import EvaluationEngine
from edgegen.design_space import YoungerGenerator
from edgegen.repository import ModelRepository
from logging import Logger
from typing import List, Dict, Any, Tuple
import uuid
import random
from edgegen.conversion import torch2tflite, nn_translation
import networkx as nx
from edgegen.design_space.architectures.younger.utils.hashing import create_graph_fingerprint
import onnx

class YoungerRandomWalk:
    def __init__(self, eval_engine:EvaluationEngine, 
                 generator:YoungerGenerator, 
                 model_repo: ModelRepository,
                 parameters: List[Dict[str, Any]],
                 input_size: Tuple[int, int, int, int], #TODO - find an elegant way to pass this
                 output_size: Any,
                 logger: Logger = None):
        self.eval_engine = eval_engine
        self.generator = generator
        self.parameters = parameters
        self.model_repo = model_repo
        self.input_size = input_size 
        self.output_size = output_size
        self.logger = logger


    def evaluate(self, arch):
        eval_result = self.eval_engine.evaluate(arch)

        num_unsatisfied = len(eval_result.unsatisfied_constraints)
        if num_unsatisfied == 0:
            model_id = str(uuid.uuid4())
            self.model_repo.save(arch, model_id)
            torch2tflite.convert(arch, self.input_size, self.model_repo.model_folder / model_id)

            if self.logger is not None:
                msg = f'Valid architecture configuration found - {model_id}'
                msg += f"\nMetrics: {eval_result.metrics}"

                msg += "\nSatisfied constraints:"
                if len(eval_result.satisfied_constraints) == 0:
                    msg += "\nNone"
                else:
                    for i in eval_result.satisfied_constraints:
                        msg += f"\n{self.eval_engine.constraint_manager.constraints[i]}"

                msg += "\n\nUnsatisfied constraints:"
                if len(eval_result.unsatisfied_constraints) == 0:
                    msg += "\nNone"
                else:
                    for i in eval_result.unsatisfied_constraints:
                        msg += f"\n{self.eval_engine.constraint_manager.constraints[i]}"

                self.logger.info(msg)

        # Return metrics and constraints
        result = eval_result.metrics
        result['num_unsatisfied'] = num_unsatisfied

        return result
    
    def run(self, max_walk_length:int, seed:int=None, hashes_of_generated_graphs:List = []) -> List[str]:
        """
        Perform multiple random walks on the graph starting from random start nodes.
        
        Args:
            graph (nx.Graph): The NetworkX graph to walk on.
            start_nodes (list): List of possible start nodes.
            end_nodes (list): List of possible end nodes.
            max_walk_length (int): Maximum length of the walk.
        
        Returns:
            list: A list of graph hashes.
        """
        if seed is not None:
            import numpy as np
            random.seed(seed)
            np.random.seed(seed)

        generator_specs = self.generator.get_input_spec()(**self.parameters)
        younger_net = self.generator.generate(generator_specs)
        graph = younger_net.super_graph
        start_nodes = younger_net.input_nodes
        end_nodes = younger_net.output_nodes

        current_node = random.choice(start_nodes)
        walk_graph = nx.DiGraph()
        
        current_node_guid = str(uuid.uuid4())
        walk_graph.add_node(current_node_guid, **graph.nodes[current_node])

        neighbors = list(graph.neighbors(current_node))
        
        for _ in range(max_walk_length - 1):  # max_walk_length - 1 because we already have the start node
            if not neighbors:  # If no neighbors, stop the walk
                break
            current_end_nodes = [n for n in neighbors if n in end_nodes]
            current_options = current_end_nodes if current_end_nodes else neighbors

            next_node = random.choice(current_options)
            next_node_guid = str(uuid.uuid4())

            if next_node in end_nodes:
                sub_walk_graph = walk_graph.copy()
                sub_walk_graph.add_node(next_node_guid, **graph.nodes[next_node])
                sub_walk_graph.add_edge(current_node_guid, next_node_guid)

                sub_walk_graph_hash = create_graph_fingerprint(sub_walk_graph)

                if sub_walk_graph_hash not in hashes_of_generated_graphs:
                    try: 
                        onnx_arch = nn_translation.networkx_to_onnx(sub_walk_graph, self.input_size, self.output_size)
                        torch_arch = nn_translation.onnx_to_pytorch(onnx_arch)
                        self.evaluate(torch_arch)
                    # Catch system exit and exception
                    except (SystemExit, Exception) as e:
                        onnx.save_model(onnx_arch, self.model_repo.model_folder / f"test.onnx")
                        inferred_model = onnx.shape_inference.infer_shapes(onnx_arch, data_prop=True)
                        inferred_shapes = {vi.name: [dim.dim_value for dim in vi.type.tensor_type.shape.dim] for vi in inferred_model.graph.value_info}
                        print(f"Error evaluating architecture: {e}")
                        print(f"Architecture: {torch_arch}")
                    hashes_of_generated_graphs.append(sub_walk_graph_hash)

                neighbors.pop(neighbors.index(next_node))
            else:
                walk_graph.add_node(next_node_guid, **graph.nodes[next_node])
                walk_graph.add_edge(current_node_guid, next_node_guid)
                current_node_guid = next_node_guid
                current_node = next_node
                neighbors = list(graph.neighbors(current_node))
        
        return hashes_of_generated_graphs
