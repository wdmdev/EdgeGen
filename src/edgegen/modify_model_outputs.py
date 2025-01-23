"""
Script used to modify the output shape of models in a specific folder to a specific shape (1, N).
This was necessary because the used setup with Edge Impulse only supports 2d outputs.
"""
import onnx
from onnx import helper, shape_inference, TensorProto
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
import uuid
import os
import glob
from tqdm import tqdm
from edgegen.conversion import torch2tflite, nn_translation


def modify_onnx_model_with_shape_inference(output_shape, input_model_path, output_dir):
    # Load the original ONNX model
    model = onnx.load(input_model_path)
    graph = model.graph

    input_shape = [dim.dim_value for dim in graph.input[0].type.tensor_type.shape.dim]

    # Get the last node in the graph
    next_to_last_node = graph.node[-2]
    last_node = graph.node.pop()

    # Add a Flatten layer
    flatten_output = "custom_final_flatten"
    flatten_node = helper.make_node(
        "Flatten",
        inputs=[next_to_last_node.output[0]],
        outputs=[flatten_output],
        name="custom_final_flatten"
    )
    graph.node.append(flatten_node)

    # Perform shape inference to calculate the shape after Flatten
    inferred_model = shape_inference.infer_shapes(model)
    inferred_graph = inferred_model.graph

    # Get the shape of the output of the Flatten layer
    inferred_shapes = {
        value_info.name: value_info.type.tensor_type.shape
        for value_info in inferred_graph.value_info
    }

    flatten_shape = [
        dim.dim_value for dim in inferred_shapes[flatten_output].dim
    ]

    # Define MatMul weights based on the Flatten output shape
    matmul_input_dim = int(np.prod(flatten_shape[1:]))
    matmul_output_dim = output_shape[-1]  # Example output dimension
    matmul_weight_name = "custom_final_matmul_W"
    matmul_output = "custom_final_matmul"
    matmul_weight = helper.make_tensor(
        name=matmul_weight_name,
        data_type=TensorProto.FLOAT,
        dims=[matmul_input_dim, matmul_output_dim],
        vals=np.random.rand(matmul_input_dim, matmul_output_dim).astype(np.float32).flatten()
    )
    graph.initializer.append(matmul_weight)

    # Add the MatMul node
    matmul_node = helper.make_node(
        "MatMul",
        inputs=[flatten_output, matmul_weight_name],
        outputs=[matmul_output],
        name="custom_final_matmul"
    )
    graph.node.append(matmul_node)

    # Define Add bias based on the MatMul output shape
    add_bias_name = "custom_final_add_B"
    add_output = "custom_final_add"
    add_bias = helper.make_tensor(
        name=add_bias_name,
        data_type=TensorProto.FLOAT,
        dims=[matmul_output_dim],
        vals=np.random.rand(matmul_output_dim).astype(np.float32).flatten()
    )
    graph.initializer.append(add_bias)

    # Add the Add node
    add_node = helper.make_node(
        "Add",
        inputs=[matmul_output, add_bias_name],
        outputs=[add_output],
        name="custom_final_add"
    )
    graph.node.append(add_node)

    # Modify the last node to take the output of the Add layer
    if last_node.op_type in ['Softmax', 'LogSoftmax']:
        for attr in last_node.attribute:
            if attr.name == 'axis':
                attr.i = 1 
                break
            
    last_node.input[0] = add_output
    graph.node.append(last_node)

    # Save the modified model
    torch_arch = nn_translation.onnx_to_pytorch(model)

    output_model_path = output_dir / Path(input_model_path).stem
    torch2tflite.convert(torch_arch, input_shape, output_model_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--folder_path", type=str,  
                        help="Path to the folder containing the models to modify",
                        default="/home/wdm/EdgeGen/output/YoungerGenerator_63d624cb-1f6d-4390-83dd-af47d15df0bb")

    args = parser.parse_args()

    name_of_input_folder = args.folder_path.split("/")[-1]
    output_dir = Path(__file__).parent.parent.parent / "output" / (f"modified_{uuid.uuid4().hex}_" + name_of_input_folder)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    
    all_tflite_files = glob.glob(os.path.join(args.folder_path, '**', '*_int8.tflite'), recursive=True)

    sucess_count = 0
    failed_count = 0
    for file_path in tqdm(all_tflite_files):
        if file_path.endswith("_int8.tflite"):
            onnx_file = file_path.split("_int8.tflite")[0] + ".onnx" 
            if os.path.exists(onnx_file):
                try:
                    modify_onnx_model_with_shape_inference((1, 10), onnx_file, output_dir)
                    sucess_count += 1
                except Exception as e:
                    failed_count += 1
                    continue
    
    print(f"Modified {sucess_count} models")
    print(f"Failed to modify {failed_count} models")

