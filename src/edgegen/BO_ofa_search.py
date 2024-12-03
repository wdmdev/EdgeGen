from ax.service.managed_loop import optimize
from edgegen.evaluation import EvaluationEngine
from edgegen.generator import ArchitectureGenerator
from edgegen.repository import ModelRepository
from logging import Logger
from typing import List, Dict, Any, Union
import uuid

class BOSearch:
    def __init__(self, eval_engine:EvaluationEngine, 
                 generator:ArchitectureGenerator, 
                 model_repo: ModelRepository,
                 parameters: List[Dict[str, Any]],
                 logger: Logger = None):
        self.eval_engine = eval_engine
        self.generator = generator
        self.parameters = parameters
        self.model_repo = model_repo
        self.logger = logger


    def evaluate(self, params):
        # Evaluate architecture
        generator_specs = self.generator.get_input_spec()(**params)
        arch = self.generator.generate(generator_specs)
        eval_result = self.eval_engine.evaluate(arch)

        num_unsatisfied = len(eval_result.unsatisfied_constraints)
        if num_unsatisfied == 0:
            model_id = str(uuid.uuid4())
            self.model_repo.save(arch, model_id)

            if self.logger is not None:
                msg = f'Valid architecture configuration found - {model_id}'
                # log satisfied and unsatisfied constraints
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

                msg += "\n\nGenerator specs:"
                msg += str(generator_specs)
                msg += "\n\n"

                self.logger.info(msg)

        # Return metrics and constraints
        result = eval_result.metrics
        result['num_unsatisfied'] = num_unsatisfied

        return result
        # return {"objective": eval_result.metrics['flash_memory'], "constraint": num_unsatisfied,
        #         "flash": eval_result.metrics['flash'], 'memory': eval_result.metrics['memory']}
    
    def run(self, outcome_constraints:Union[List[str], None]=None):
        satisfied_const = 'num_unsatisfied <= 0'
        if not any([o for o in outcome_constraints if o.lower() == satisfied_const]):
            outcome_constraints.append(satisfied_const)

        optimize(
            self.parameters,
            evaluation_function=self.evaluate,
            objective_name="objective",
            minimize=True,
            outcome_constraints=outcome_constraints)