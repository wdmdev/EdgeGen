from edgegen.evaluation import EvaluationEngine
from edgegen.design_space import YoungerSelector
from edgegen.repository import ModelRepository
from logging import Logger
from typing import List, Dict, Any, Tuple
import uuid
import random
from edgegen.conversion import torch2tflite, nn_translation
from edgegen.design_space.architectures.younger.utils.hashing import create_graph_fingerprint
from tqdm import tqdm
import traceback

class YoungerSelection:
    def __init__(self, eval_engine:EvaluationEngine, 
                 generator:YoungerSelector, 
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
    
    def run(self, seed:int=None, hashes_of_selected_graphs:List = []) -> List[str]:
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
        younger_graphs = self.generator.generate(generator_specs)

        for graph in tqdm(younger_graphs, total=sum(1 for _ in younger_graphs)):
            graph_hash = create_graph_fingerprint(graph)

            if graph_hash not in hashes_of_selected_graphs:
                onnx_arch = nn_translation.networkx_to_onnx(graph, self.input_size, self.output_size)
                torch_arch = nn_translation.onnx_to_pytorch(onnx_arch)
                try:
                    self.evaluate(torch_arch)
                except SystemExit as e:
                    self.logger.error(f"""{self.__class__.__name__} System Exit evaluating architecture\n
                                      code: {e.code}\n{traceback.format_stack()}""")
                    continue
                except Exception as e:
                    self.logger.error(f"""{self.__class__.__name__} Exception evaluating architecture\n{e}
                    \n{traceback.format_exc()}""")
                    continue
                hashes_of_selected_graphs.append(graph_hash)