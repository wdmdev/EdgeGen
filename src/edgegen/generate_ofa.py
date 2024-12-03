#TODO - Change to ArgumentParser and possibility to use configs

if __name__ == "__main__":
    import os
    import random
    import uuid
    from pathlib import Path
    from edgegen.generator import OFAProxylessNasGenerator, OFAProxylessNasInputSpec
    from edgegen.evaluation import ConstraintManager, PyTorchMemoryConstraint, PyTorchFlashConstraint
    from edgegen.evaluation.eval_engine import EvaluationEngine
    from edgegen.repository import PytorchModelRepository
    from edgegen import Bytes
    from edgegen.utils.logging import get_logger


    # Create test constraint manager and constraints
    input_size = (1, 3, 128, 128)

    # Create test evaluation engine
    metrics = {
        "model_param_type": lambda arch: set(p.dtype for p in arch.parameters()),
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

    # Create test architecture generator
    seed = random.randint(0, 10000)
    generator_spec = OFAProxylessNasInputSpec(
        n_classes=10,
        bn_param=(0.1, 1e-3),
        dropout_rate=0.1,
        base_stage_width=None,
        width_mult_list=[0.5],
        ks_list=[3],
        expand_ratio_list=[9],
        depth_list=[2],
        no_mix_layer=False,
        seed=random.randint(0, 10000)
    )
    archGenerator = OFAProxylessNasGenerator(generator_spec)

    # Create test model repository
    output_folder = Path(__file__).parent.parent / 'output'/ (archGenerator.__class__.__name__ + '_' + str(uuid.uuid4()))
    os.makedirs(output_folder, exist_ok=True)
    modelRepo = PytorchModelRepository(model_folder=output_folder)


    log_path_prefix = archGenerator.__class__.__name__ + 'seed' + str(seed)
    logger = get_logger(log_dir=output_folder, log_path_prefix=log_path_prefix)
    msg = f'Testing with seed: {seed}'

    # Test evaluation
    arch = archGenerator.generate()
    modelRepo.save(arch)
    eval_result = eval_engine.evaluate(arch)

    # log satisfied and unsatisfied constraints
    msg += f"\nMetrics: {eval_result.metrics}"

    msg += "\nSatisfied constraints:"
    for i in eval_result.satisfied_constraints:
        msg += f"\n{constraints[i]}"
    
    msg += "\n\nUnsatisfied constraints:"
    for i in eval_result.unsatisfied_constraints:
        msg += f"\n{constraints[i]}"

    logger.info(msg)