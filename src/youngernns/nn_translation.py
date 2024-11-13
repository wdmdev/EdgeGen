import os
from argparse import ArgumentParser
import ast
import onnx
from onnx import helper
from onnx.shape_inference import infer_shapes
import tensorflow as tf
import numpy as np
from typing import Union
from onnx.onnx_ml_pb2 import ModelProto
from onnx.checker import check_model
import onnx2tf
import tf_keras
import networkx as nx
from youngernns.nn_query import select_networks_with_operators

def __convert_attribute_to_type(attr_value, attr_type):
    supported_types = {
        onnx.defs.OpSchema.AttrType.FLOAT,
        onnx.defs.OpSchema.AttrType.INT,
        onnx.defs.OpSchema.AttrType.STRING,
        onnx.defs.OpSchema.AttrType.GRAPH,
        onnx.defs.OpSchema.AttrType.TENSOR,
        onnx.defs.OpSchema.AttrType.FLOATS,
        onnx.defs.OpSchema.AttrType.INTS,
        onnx.defs.OpSchema.AttrType.STRINGS,
    }

    trans_method = {
        onnx.defs.OpSchema.AttrType.FLOAT: lambda x: float(x),
        onnx.defs.OpSchema.AttrType.FLOATS: lambda x: ast.literal_eval(x),
        onnx.defs.OpSchema.AttrType.INT: lambda x: int(x),
        onnx.defs.OpSchema.AttrType.INTS: lambda x: [int(i) for i in ast.literal_eval(x)],
        onnx.defs.OpSchema.AttrType.STRING: lambda x: str(x),
        onnx.defs.OpSchema.AttrType.STRINGS: lambda x: [str(s) for s in ast.literal_eval(x)],

        onnx.defs.OpSchema.AttrType.GRAPH: lambda x: f"node_{x}",

        onnx.defs.OpSchema.AttrType.TENSOR: lambda x: onnx.helper.make_tensor(**ast.literal_eval(x)),
    }

    if attr_type in supported_types:
        value = trans_method[attr_type](attr_value)
    else:
        raise ValueError(f"The attribute type {attr_type} is not supported for conversion.")
    
    return value

def pre_shape_inference_op_inputs(node, attrs, node_attrs, input_shape):
    input = None
    input_name = None

    if attrs['operator']['op_type'] == 'Reshape':
        batch_size = input_shape[0]
        target_shape = [batch_size, -1]
        input_name = f"Reshape_shape_{node}"
        input = helper.make_tensor(
            input_name,
            onnx.TensorProto.INT64,
            dims=(len(target_shape),),
            vals=target_shape
        )
    elif attrs['operator']['op_type'] == 'Range':
        # Define scalar values for start, limit, and delta
        start_value = np.array(0, dtype=np.int64)
        limit_value = np.array(10, dtype=np.int64)
        delta_value = np.array(1, dtype=np.int64)

        # Create unique names for the tensors
        start_name = f"Range_start_{node}"
        limit_name = f"Range_limit_{node}"
        delta_name = f"Range_delta_{node}"

        # Create tensors for each scalar value
        start_tensor = helper.make_tensor(
            name=start_name,
            data_type=onnx.TensorProto.INT64,
            dims=(),
            vals=start_value.flatten()
        )
        limit_tensor = helper.make_tensor(
            name=limit_name,
            data_type=onnx.TensorProto.INT64,
            dims=(),
            vals=limit_value.flatten()
        )
        delta_tensor = helper.make_tensor(
            name=delta_name,
            data_type=onnx.TensorProto.INT64,
            dims=(),
            vals=delta_value.flatten()
        )
        input = [start_tensor, limit_tensor, delta_tensor]
        input_name = [start_name, limit_name, delta_name]
    elif attrs['operator']['op_type'] == 'Unsqueeze':
        axes = node_attrs.get('axes', None)
        if axes is None:
            axes = attrs['attributes'].get('axes', None)
            if axes:
                axes = axes['value']
            else:
                raise ValueError("Axes for Unsqueeze operation must be provided.")
        input_name = f"Unsqueeze_axes_{node}"
        input = helper.make_tensor(
            input_name,
            onnx.TensorProto.INT64,
            dims=(len(axes),),
            vals=axes
        )
    elif attrs['operator']['op_type'] == 'Gather':
        indices = np.array([0, 1, 2, 3], dtype=np.int64)
        input_name = f"Gather_indices_{node}"
        input = helper.make_tensor(
            input_name,
            onnx.TensorProto.INT64,
            dims=(len(indices),),
            vals=indices
        )

    return input, input_name

def post_shape_inference_op_inputs(node, attrs, node_attrs, shape_tracker, input_names):
    input = None
    input_name = None

    if attrs['operator']['op_type'] == "Conv":
        num_filters = 3  # You may adjust this as needed
        kernel_shape = node_attrs['kernel_shape']
        group = node_attrs['group']
        num_channels = shape_tracker[input_names[0]][1]
        weight_shape = [num_filters, num_channels // group, *kernel_shape]

        # Create weight tensor
        input_name = f"Conv_weights_{node}"
        weights = np.random.rand(*weight_shape).astype(np.float32)
        input = helper.make_tensor(
            name=input_name,
            data_type=onnx.TensorProto.FLOAT,
            dims=weight_shape,
            vals=weights.flatten(),
        )
    elif attrs['operator']['op_type'] == 'MatMul':
        input_dim = shape_tracker[input_names[0]][1]
        output_dim = 2 # choose yourself
        y_values = np.random.rand(input_dim, output_dim).astype(np.float32)
        # Create the tensor for Y
        input_name = f"Matmul_Y_{node}"
        input = helper.make_tensor(
            name=input_name,
            data_type=onnx.TensorProto.FLOAT,
            dims=(input_dim, output_dim),
            vals=y_values.flatten().tolist()
        )
    elif attrs['operator']['op_type'] == 'Add':
        add_dim = shape_tracker[input_names[0]]
        add_values = np.random.rand(*add_dim).astype(np.float32)
        # Create the tensor for Y
        input_name = f"Add_Y_{node}"
        input = helper.make_tensor(
            name=input_name,
            data_type=onnx.TensorProto.FLOAT,
            dims=add_dim,
            vals=add_values.flatten().tolist()
        )
        
    return input, input_name


def networkx_to_onnx(nx_graph, input_shape, output_shape):
    nodes = list(nx_graph.nodes(data=True))
    edges = list(nx_graph.edges(data=True))

    # Create ONNX nodes
    onnx_nodes = []
    initializers = []
    shape_tracker = {"data": input_shape}  # Initialize with input shape

    for node, attrs in nodes:
        node_attrs = {}

        # Define inputs for this node based on incoming edges
        input_names = [f"node_{u}" for u, v, _ in edges if v == node]
        if input_names == []:
            input_names = ["data"]

        for att, (att_type, val) in attrs['features']['attributes'].items():
            if ast.literal_eval(val) is not None:
                if "NOTSET" not in val:
                    node_attrs[att] = __convert_attribute_to_type(val, att_type)

        pre_shape_op_input, pre_shape_op_input_name = pre_shape_inference_op_inputs(node, attrs, node_attrs, input_shape)
        if pre_shape_op_input is not None:
            if isinstance(pre_shape_op_input_name, list):
                input_names.extend(pre_shape_op_input_name)
                initializers.extend(pre_shape_op_input)
            else:
                input_names.append(pre_shape_op_input_name)
                initializers.append(pre_shape_op_input)
        
        # Infer shapes with a temporary ONNX model
        name = f"node_{node}"
        onnx_node = helper.make_node(
            attrs['operator']['op_type'],
            inputs=input_names,
            outputs=[name],
            name=name,
            **node_attrs,
        )
        onnx_nodes.append(onnx_node)
        temp_graph = helper.make_graph(
            onnx_nodes,
            "graph",
            inputs=[helper.make_tensor_value_info("data", onnx.TensorProto.FLOAT, input_shape)],
            outputs=[helper.make_tensor_value_info(onnx_nodes[-1].name, onnx.TensorProto.FLOAT, output_shape)],
            initializer=initializers
        )
        temp_model = helper.make_model(temp_graph)
        inferred_model = infer_shapes(temp_model)
        
        # Update the shape tracker with inferred shapes
        inferred_shapes = {vi.name: [dim.dim_value for dim in vi.type.tensor_type.shape.dim] for vi in inferred_model.graph.value_info}
        shape_tracker.update(inferred_shapes)

        post_shape_op_input, post_shape_op_input_name = post_shape_inference_op_inputs(node, attrs, node_attrs, shape_tracker, input_names)
        if post_shape_op_input is not None:
            if isinstance(post_shape_op_input_name, list):
                initializers.extend(post_shape_op_input)
                input_names.extend(post_shape_op_input_name)
            else:
                initializers.append(post_shape_op_input)
                input_names.append(post_shape_op_input_name)

        onnx_node = helper.make_node(
            attrs['operator']['op_type'],
            inputs=input_names,
            outputs=[name],
            name=name,
            **node_attrs,
        )
        onnx_nodes[-1] = onnx_node
    
    # Define inputs and outputs
    inputs = [helper.make_tensor_value_info("data", onnx.TensorProto.FLOAT, input_shape)]
    outputs = [helper.make_tensor_value_info(onnx_nodes[-1].name, onnx.TensorProto.FLOAT, output_shape)]

    # Create the ONNX graph
    onnx_graph = helper.make_graph(
        onnx_nodes,
        "graph_name",
        inputs=inputs,
        outputs=outputs,
        initializer=initializers,
    )

    # Build the ONNX model
    onnx_model = helper.make_model(onnx_graph)
    # infered_model = infer_shapes(onnx_model) 
    # return infered_model
    return onnx_model

def onnx_to_tf(onnx_model: Union[ModelProto, str], output_path: str) -> object:
    if isinstance(onnx_model, str):
        onnx_model = onnx.load(onnx_model)
    
    shaped_onnx_model = infer_shapes(onnx_model)

    check_model(shaped_onnx_model)

    onnx2tf.convert(onnx_graph=shaped_onnx_model, 
                               batch_size=1,
                               keep_shape_absolutely_input_names=["data"],
                               output_folder_path=output_path,
                               copy_onnx_input_output_names_to_tflite = True,
                               non_verbose=True)
    
    return tf.saved_model.load(output_path)

def tf_to_networkx(tf_model: tf_keras.Model) -> nx.DiGraph:
    concrete_func = tf_model.signatures['serving_default']
    tf_graph =concrete_func.graph.as_graph_def()
    function_library = tf_graph.library
    function_defs = [f for f in function_library.function]
    node_def_attrs = [{'op': n.op, 'input': n.input, "output": n.name} for n in function_defs[0].node_def]

    # Create a NetworkX graph
    nx_graph = nx.DiGraph()
    for node in node_def_attrs:
        nx_graph.add_node(node['output'], operator=node['op'])

    return nx_graph


if __name__ == "__main__":
    from tqdm import tqdm
    from pathlib import Path
    from youngernns.utils.logging import get_logger
    from youngernns.utils.io import save_pickle
    from youngernns.data.network import count_networks, load_networks
    from youngernns.nn_query import select_networks_with_operators

    logger = get_logger(log_path_prefix=os.path.basename(__file__).replace(".py", ""))

    parser = ArgumentParser()
    parser.add_argument("--data_path", help="Path to folder containing network data in the Younger Dataset format.", 
                        type=str, required=True)
    parser.add_argument("--operators", help="""List of operators to select networks for conversion based on.\n
                        If specified only networks with a least one of the operators will be selected for conversion.""",
                        required=False, type=str, nargs='+', default=None)
    args = parser.parse_args()

    print(f"Starting conversion of network graphs in {args.data_path} to TensorFlow(TF) format.")
    print(f"No existing files will be changed.")
    print(f"A new tf_graph.pkl file will be created for each network in the data path.")
    print(f"This tf_graph.pkl will be a NetworkX graph object with operator and attribute information following the TF format.")

    data_path = Path(args.data_path)
    num_networks = count_networks(args.data_path)

    if args.operators is not None:
        networks_enumerable = select_networks_with_operators(data_path=data_path, operators=args.operators)
    else:
        networks_enumerable = load_networks(data_path=data_path)

    selected_network_count = 0
    for network in tqdm(networks_enumerable, total=num_networks):
        network_folder = network.data_path
        if network_folder is not None:
            tf_graph_filename = os.path.join(network_folder, 'tf_graph.pkl')
            # Check that the network_folder does not contain a tf_graph.pkl file
            if not os.path.exists(tf_graph_filename):
                try:
                    onnx_model = networkx_to_onnx(network.graph, input_shape=(1,128,128,3), output_shape=(1,2))
                    tf_model = onnx_to_tf(onnx_model, output_path=os.path.join(network_folder, 'tf_model'))
                    tf_graph = tf_to_networkx(tf_model)
                    save_pickle(tf_graph, tf_graph_filename)
                    selected_network_count += 1
                    logger.info(f"Successfully converted network {network_folder} to TensorFlow format.")
                except Exception as e:
                    logger.error(f"Error converting network {network_folder} to TensorFlow format: {e}", exc_info=True)
                except SystemExit as e:
                    logger.error(f"Error converting network {network_folder} to TensorFlow format: {e}", exc_info=True)
    
    success_message = f"Successfully converted {selected_network_count} networks to TensorFlow format."
    logger.info(success_message)
    print(success_message)
