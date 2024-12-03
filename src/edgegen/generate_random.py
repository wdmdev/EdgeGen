#TODO - Change to ArgumentParser and possibility to use configs

if __name__ == "__main__":
    import os
    import random
    from pathlib import Path
    from edgegen.generator import RandomPytorchArchitectureGenerator, RandomPytorchArchitectureSpec
    from edgegen.evaluation import ConstraintManager, PyTorchMemoryConstraint, PyTorchFlashConstraint
    from edgegen.evaluation.eval_engine import EvaluationEngine
    from edgegen.repository import PytorchModelRepository
    from edgegen import Bytes
    from logging import getLogger, StreamHandler, Formatter, DEBUG

    # Create test constraint manager and constraints
    input_size = (3, 128, 128)
    constraints = [
        PyTorchMemoryConstraint(input_size, max_memory_limit=Bytes.from_KB(600), quant_size=4),
        PyTorchFlashConstraint(max_flash_limit=Bytes.from_KB(900), quant_size=4),
    ]
    constraint_manager = ConstraintManager(constraints=constraints)

    # Create test evaluation engine
    metrics = {
        "model_param_type": lambda arch: set(p.dtype for p in arch.parameters()),
    }

    eval_engine = EvaluationEngine(constraint_manager=constraint_manager, metrics=metrics)

    # Create test architecture generator
    generator_spec = RandomPytorchArchitectureSpec(
        kernel_sizes=[3, 5],
        strides=[1, 2],
        min_depth=1,
        max_depth=3,
        min_filters=16,
        max_filters=128,
        min_fc_layers=1,
        max_fc_layers=3,
        min_neurons=64,
        max_neurons=512,
        num_classes=10,
        input_channels=3,
        input_size=(3, 128, 128),
        seed=random.randint(0, 10000)
    )
    archGenerator = RandomPytorchArchitectureGenerator(generator_spec)

    # Create test model repository
    model_folder = Path(__file__).parent / 'output' / 'models'
    os.makedirs(model_folder, exist_ok=True)
    modelRepo = PytorchModelRepository(model_folder=model_folder)

    # Create test logger and set the log output to be in ../output/logs/EdgeGen.log
    logger = getLogger()
    logger.setLevel(DEBUG)
    log_folder = Path(__file__).parent / 'output' / 'logs'
    os.makedirs(log_folder, exist_ok=True)
    log_file = log_folder / 'EdgeGen.log'
    handler = StreamHandler(log_file)
    handler.setFormatter(Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.info(f'Testing with seed: {generator_spec.seed}')

    # Test evaluation
    arch = archGenerator.generate()
    modelRepo.save(arch)
    eval_result = eval_engine.evaluate(arch)
    print(eval_result)