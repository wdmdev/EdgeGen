#TODO - Change to ArgumentParser and possibility to use configs

if __name__ == "__main__":
    import os
    import uuid
    from pathlib import Path
    from edgegen.generator import OFAProxylessNasGenerator, MicroNetGenerator
    from edgegen.evaluation import ConstraintManager, PyTorchMemoryConstraint, PyTorchFlashConstraint
    from edgegen.evaluation.eval_engine import EvaluationEngine
    from edgegen.repository import PytorchModelRepository
    from edgegen import Bytes
    from edgegen.utils import get_logger
    from edgegen.evaluation.utils import estimate_torch_flash, estimate_torch_mem
    from edgegen.search.ax_bo import BOSearch
    from argparse import ArgumentParser

    #add argument parser and epochs as argument
    parser = ArgumentParser()
    parser.add_argument("--BO_epochs", type=int, default=20)

    args = parser.parse_args()

    # Create test constraint manager and constraints
    input_size = (1, 3, 128, 128)

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

    eval_engine = EvaluationEngine(constraint_manager=constraint_manager, metrics=metrics)

    archGenerator = OFAProxylessNasGenerator()
    # archGenerator = MicroNetGenerator()

    output_folder = Path(__file__).parent.parent.parent / 'output'/ (archGenerator.__class__.__name__ + '_' + str(uuid.uuid4()))
    os.makedirs(output_folder, exist_ok=True)
    model_repo = PytorchModelRepository(model_folder=output_folder)

    logger = get_logger(log_dir=output_folder, log_path_prefix='', name="BOSearch")

    # OFAProxylessNasGenerator params
    params = [
        {"name": "n_classes", "type": "range", "bounds": [2, 10], "value_type": "int"},
        {"name": "bn_momentum", "type": "fixed", "value": 0.1},
        {"name": "bn_eps", "type": "fixed", "value": 1e-3},
        {"name": "dropout_rate", "type": "fixed", "value": 0.1},
        {"name": "base_stage_width", "type": "choice", "values": ['google', '']},
        {"name": "width_mult_list", "type": "range", "bounds": [0.1, 1.0]},
        {"name": "ks_list", "type": "choice", "values": list(range(3, 11, 2)), "value_type": "int"},
        {"name": "expand_ratio_list", "type": "range", "bounds": [2,10], "value_type": "int"},
        {"name": "depth_list", "type": "range", "bounds": [2, 150], "value_type": "int"},
        {"name": "no_mix_layer", "type": "choice", "values": [True, False]},
    ]

    # MicronetGenerator params
    # params = [
    #     {"name": "input_size", "type": "range", "bounds": [32, 256], "value_type": "int"},
    #     {"name": "num_classes", "type": "range", "bounds": [2, 1000], "value_type": "int"},
    #     {"name": "teacher", "type": "choice", "values": [True, False], "value_type": "bool"},
    #     {"name": "block", "type": "choice", "values": ["block1", "block2", "block3"], "value_type": "str"},
    #     {"name": "stem_ch", "type": "range", "bounds": [16, 128], "value_type": "int"},
    #     {"name": "stem_dilation", "type": "range", "bounds": [1, 4], "value_type": "int"},
    #     {"name": "dropout_rate", "type": "range", "bounds": [0.0, 0.5], "value_type": "float"},
    #     {"name": "activation_cfg.module", "type": "choice", "values": ["relu", "swish", "mish"], "value_type": "str"},
    #     {"name": "activation_cfg.act_max", "type": "range", "bounds": [0.0, 10.0], "value_type": "float"},
    #     {"name": "activation_cfg.act_bias", "type": "choice", "values": [True, False], "value_type": "bool"},
    #     {"name": "activation_cfg.reduction", "type": "range", "bounds": [1, 16], "value_type": "int"},
    # ]

    outcome_constraints = [
        f'flash <= {constraints[1].max_flash_limit.size}',
        f'memory <= {constraints[0].max_memory_limit.size}',
    ]

    searcher = BOSearch(eval_engine, 
                        archGenerator,
                        model_repo,
                        params,
                        input_size,
                        logger)
    searcher.run(outcome_constraints, epochs=args.BO_epochs)