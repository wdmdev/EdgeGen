import onnx
from collections import defaultdict
from onnx import numpy_helper, shape_inference
import pandas as pd

def extract_model_features(model_path):
    # Load the ONNX model
    model = onnx.load(model_path)
    # Perform shape inference to determine tensor shapes
    inferred_model = shape_inference.infer_shapes(model)
    graph = inferred_model.graph

    # Initialize feature containers
    layer_types = defaultdict(int)
    parameter_counts = defaultdict(int)
    input_output_shapes = {}
    activation_functions = set()

    # Create a dictionary to map initializer names to their numpy arrays
    initializer_dict = {init.name: numpy_helper.to_array(init) for init in graph.initializer}

    # Iterate over the nodes in the graph
    for node in graph.node:
        # Count layer types
        layer_types[node.op_type] += 1

        # Collect input/output shapes
        input_shapes = []
        output_shapes = []

        for input_name in node.input:
            # Check if the input is an initializer (i.e., a parameter)
            if input_name in initializer_dict:
                param_array = initializer_dict[input_name]
                parameter_counts[node.name] += param_array.size
            else:
                # If not an initializer, it's an intermediate tensor; get its shape
                for value_info in graph.value_info:
                    if value_info.name == input_name:
                        shape = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
                        input_shapes.append((input_name, shape))

        for output_name in node.output:
            for value_info in graph.value_info:
                if value_info.name == output_name:
                    shape = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
                    output_shapes.append((output_name, shape))

        input_output_shapes[node.name] = {'inputs': input_shapes, 'outputs': output_shapes}

        # Identify activation functions
        if node.op_type in ['Relu', 'Sigmoid', 'Softmax']:
            activation_functions.add(node.op_type)

    # Calculate total number of layers
    total_layers = sum(layer_types.values())

    # Calculate layer type distribution
    layer_type_distribution = {k: v / total_layers for k, v in layer_types.items()}

    # Compile features into a dictionary
    features = {
        'Number of Layers': total_layers,
        'Layer Types Distribution': layer_type_distribution,
        'Parameter Counts': parameter_counts,
        'Input/Output Dimensions': input_output_shapes,
        'Activation Functions': list(activation_functions)
    }

    return features

def flatten_features(features):
    # Flatten `Layer Types Distribution`
    flattened = {}
    for layer_type, proportion in features.get('Layer Types Distribution', {}).items():
        flattened[f'Layer Type {layer_type} Proportion'] = proportion

    # Flatten `Parameter Counts`
    flattened['Toal Parameter Count'] = sum(features.get('Parameter Counts', {}).values())

    # Average Parameter Count per Layer
    flattened['Average Parameter Count per Layer'] = (
        flattened['Toal Parameter Count'] / features.get('Number of Layers', 1)
    )

    # Aggregate Input/Output Dimensions
    io_shapes = features.get('Input/Output Dimensions', {})
    input_sizes = []
    output_sizes = []

    for node_name, io in io_shapes.items():
        inputs = io['inputs']
        if len(inputs) > 0:
            input_sizes.extend([sum(shape[1:]) for _, shape in inputs])
        else:
            input_sizes.append(0)
        
        outputs = io['outputs']
        if len(outputs) > 0:
            output_sizes.extend([sum(shape[1:]) for _, shape in outputs])
        else:
            output_sizes.append(0)

    avg_input_size = sum(input_sizes) / len(input_sizes) if input_sizes else 0
    avg_output_size = sum(output_sizes) / len(output_sizes) if output_sizes else 0
    flattened['Average IO Size'] = (avg_input_size + avg_output_size) / 2

    # Flatten Activation Functions
    for func in ['Relu', 'Sigmoid', 'Softmax']:
        flattened[f'Activation {func}'] = int(func in features.get('Activation Functions', []))

    # Add scalar values
    flattened['Number of Layers'] = features.get('Number of Layers', 0)
    flattened['RAM (kB)'] = features.get('RAM (kB)', 0)
    flattened['Flash (kB)'] = features.get('Flash (kB)', 0)
    flattened['Latency (ms)'] = features.get('Latency (ms)', 0)
    flattened['Error (MAE)'] = features.get('Error (MAE)', 0)
    flattened['Model Name'] = features.get('Model Name', '')

    return flattened

def save_features_to_csv(features_list, csv_filename):
    # Flatten features
    flattened_features_list = [flatten_features(features) for features in features_list]
    
    # Get all unique keys for header
    all_keys = set()
    for flattened_features in flattened_features_list:
        all_keys.update(flattened_features.keys())

    headers = list(all_keys) 
    headers.remove('Model Name')
    headers.insert(0, 'Model Name')

    # Put target variables last
    headers.remove('Flash (kB)')
    headers.append('Flash (kB)')
    headers.remove('RAM (kB)')
    headers.append('RAM (kB)')
    headers.remove('Latency (ms)')
    headers.append('Latency (ms)')
    headers.remove('Error (MAE)')
    headers.append('Error (MAE)')

    df = pd.DataFrame(flattened_features_list, columns=headers)
    df.fillna(0, inplace=True)
    df.to_csv(csv_filename, index=False, sep=',', float_format='%.2f', encoding='utf-8', decimal='.')

if __name__ == '__main__':
    import glob
    import pandas as pd
    from argparse import ArgumentParser
    import numpy as np
    from pathlib import Path
    import uuid
    import os
    from tqdm import tqdm

    from edgegen.utils import get_logger


    parser = ArgumentParser()
    # parser.add_argument('--csv_path', type=str, required=True)
    # parser.add_argument('--model_folder_path', type=str, required=True)
    parser.add_argument('--csv_path', type=str, default='/home/wdm/EdgeGen/data/surrogate/Edge_Impulse_NUCLEO_L4R5ZI.csv')
    parser.add_argument('--model_folder_path', type=str, default='/home/wdm/EdgeGen/output')

    args = parser.parse_args()

    run_id = uuid.uuid4()
    output_dir = Path(__file__).parent.parent.parent / 'data' / 'surrogate'
    logger = get_logger(log_dir=output_dir, log_path_prefix=str(run_id), name=f'extract_features_{run_id}')

    # Extract model names from measurement csv
    df = pd.read_csv(args.csv_path)
    model_names = df['Model Directory'].apply(lambda x: x.split('/')[2]).unique()
    flash_target = df['Flash (kB)'].astype(np.float32)
    ram_target = df['RAM (kB)'].astype(np.float32)
    latency_target = df['average exec time (ms)'].astype(np.float32)
    error_target = df['average error (MAE)'].astype(np.float32)

    model_measurements = zip(
        model_names,
        flash_target,
        ram_target,
        latency_target,
        error_target
    )

    features_list = []
    
    for (m, flash, ram, latency, err) in tqdm(list(model_measurements)):
        onnx_model_path = os.path.join(args.model_folder_path , '**' , (m + '.onnx'))
        model_paths = glob.glob(onnx_model_path, recursive=True)

        if len(model_paths) == 1:
            features = extract_model_features(model_paths[0])
            features['Model Name'] = m
            features['Flash (kB)'] = flash
            features['RAM (kB)'] = ram
            features['Latency (ms)'] = latency
            features['Error (MAE)'] = err
            features_list.append(features)
        elif len(model_paths) > 1:
            logger.error(f'Multiple models found for {m} at path {onnx_model_path}')
            logger.error(f'Found models: {model_paths}')
        else:
            logger.error(f'Model {m} not found at path {onnx_model_path}')

    # Save all features to a CSV file
    save_features_to_csv(features_list, output_dir / f'{run_id}_features.csv')