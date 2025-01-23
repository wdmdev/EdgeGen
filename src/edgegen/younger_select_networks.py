
if __name__ == '__main__':
    import os
    import uuid
    from tqdm import tqdm
    from pathlib import Path
    from edgegen.design_space import YoungerSelector
    from edgegen.evaluation import ConstraintManager, PyTorchMemoryConstraint, PyTorchFlashConstraint
    from edgegen.evaluation.eval_engine import EvaluationEngine
    from edgegen.repository import PytorchModelRepository
    from edgegen import Bytes
    from edgegen.utils import get_logger
    from edgegen.evaluation.utils import estimate_torch_flash, estimate_torch_mem
    from edgegen.search_strategy.younger_selection import YoungerSelection
    from argparse import ArgumentParser
    
    #add argument parser and epochs as argument
    parser = ArgumentParser()
    parser.add_argument("--valid_ops", nargs='+', type=str, default=None)
    parser.add_argument("--valid_input_ops", nargs='+', type=str, default=None)
    parser.add_argument("--valid_output_ops", nargs='+', type=str, default=None)
    parser.add_argument("--hash_nodes", action='store_true', default=True)
    
    args = parser.parse_args()
    
    # Create test constraint manager and constraints
    input_size = (1, 3, 128, 128)
    output_size = (1, 10) # 10 classes
    
    # Create test evaluation engine
    metrics = {
        "flash": lambda arch: estimate_torch_flash(arch, quant_size=1).size, # estimated int8 flash
        "memory": lambda arch: estimate_torch_mem(arch, input_size, quant_size=1).size, # estimated int8 mem
        "objective": lambda arch: estimate_torch_flash(arch, quant_size=1).size + estimate_torch_mem(arch, input_size, quant_size=1).size,
    }
    
    constraints = [
        PyTorchMemoryConstraint(
            name = "Memory Constraint",
            description = "RAM <= 550 KB",
            input_size=input_size, max_memory_limit=Bytes.from_KB(550), quant_size=1),
        PyTorchFlashConstraint(
            name = "Flash Constraint",
            description = "Flash <= 900 KB",
            max_flash_limit=Bytes.from_KB(900), quant_size=1),
    ]
    constraint_manager = ConstraintManager(constraints=constraints)
    
    eval_engine:EvaluationEngine = EvaluationEngine(constraint_manager=constraint_manager, metrics=metrics)
    
    archSelector = YoungerSelector()
    
    output_folder = Path(__file__).parent.parent.parent / 'output'/ (archSelector.__class__.__name__ + '_' + str(uuid.uuid4()))
    print(f"Saving selected networks to {output_folder}")
    os.makedirs(output_folder, exist_ok=True)
    model_repo = PytorchModelRepository(model_folder=output_folder)
    
    logger = get_logger(log_dir=output_folder, log_path_prefix='', name=archSelector.__class__.__name__)
    
    params = {
        'valid_ops': args.valid_ops,
        'valid_input_ops': args.valid_input_ops,
        'valid_output_ops': args.valid_output_ops,
        'hash_nodes': args.hash_nodes,
    }
    
    searcher = YoungerSelection(eval_engine, 
                        archSelector,
                        model_repo,
                        params,
                        input_size,
                        output_size,
                        logger)
    
    hashes_of_generated_graphs = []

    searcher.run(hashes_of_selected_graphs=hashes_of_generated_graphs)
    
    print(f"Generated {len(hashes_of_generated_graphs)} unique graphs.")