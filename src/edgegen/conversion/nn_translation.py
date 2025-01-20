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
from onnx2pytorch import ConvertModel as ONNX2PytorchConvert
from google.protobuf.json_format import MessageToDict
from edgegen.design_space.architectures.younger.utils.hashing import hash_node

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

def pre_shape_inference_op_inputs(node, attrs, node_attrs, input_shape, output_nodes):
    input = None
    input_name = None

    if attrs['operator']['op_type'] == 'Reshape':
        batch_size = input_shape[0]
        if ((output_nodes[0]['operator']['op_type'] == 'Conv') &
        (input_shape[1] > input_shape[3] < input_shape[2])):
            target_shape = [batch_size, input_shape[3], input_shape[1], input_shape[2]]
        else:
            target_shape = [batch_size, -1]
        input_name = f"Reshape_{node}"
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
                axes = [0] # Default value
        input_name = f"Unsqueeze_{node}"
        input = helper.make_tensor(
            input_name,
            onnx.TensorProto.INT64,
            dims=(len(axes),),
            vals=axes
        )
    elif attrs['operator']['op_type'] == 'Gather':
        indices = np.array([0, 1, 2, 3], dtype=np.int64)
        input_name = f"Gather_{node}"
        input = helper.make_tensor(
            input_name,
            onnx.TensorProto.INT64,
            dims=(len(indices),),
            vals=indices
        )
    elif attrs['operator']['op_type'] == 'BatchNormalization':
        # Extract input dimensions for scale, bias, mean, and variance
        input_dim = input_shape[1]
        scale_values = np.random.rand(input_dim).astype(np.float32)
        bias_values = np.random.rand(input_dim).astype(np.float32)
        mean_values = np.random.rand(input_dim).astype(np.float32)
        variance_values = np.random.rand(input_dim).astype(np.float32)

        # Create the tensors for scale, bias, mean, and variance
        scale_name = f"BatchNorm_scale_{node}"
        bias_name = f"BatchNorm_bias_{node}"
        mean_name = f"BatchNorm_mean_{node}"
        variance_name = f"BatchNorm_variance_{node}"

        scale_tensor = helper.make_tensor(
            name=scale_name,
            data_type=onnx.TensorProto.FLOAT,
            dims=(input_dim,),
            vals=scale_values.flatten()
        )
        bias_tensor = helper.make_tensor(
            name=bias_name,
            data_type=onnx.TensorProto.FLOAT,
            dims=(input_dim,),
            vals=bias_values.flatten()
        )
        mean_tensor = helper.make_tensor(
            name=mean_name,
            data_type=onnx.TensorProto.FLOAT,
            dims=(input_dim,),
            vals=mean_values.flatten()
        )
        variance_tensor = helper.make_tensor(
            name=variance_name,
            data_type=onnx.TensorProto.FLOAT,
            dims=(input_dim,),
            vals=variance_values.flatten()
        )

        # Return the tensors as inputs for the BatchNormalization node
        input = [scale_tensor, bias_tensor, mean_tensor, variance_tensor]
        input_name = [scale_name, bias_name, mean_name, variance_name]

    return input, input_name

def post_shape_inference_op_inputs(node, attrs, node_attrs, shape_tracker):
    input = None
    input_name = None
    input_shape = list(shape_tracker.values())[-1]

    if attrs['operator']['op_type'] == "Conv":

        # Make sure kernel_shape is matching input_shape (B, C, H, W)
        if 'kernel_shape' not in node_attrs:
            kernel_shape = (3, 3) if input_shape[-1] >= 3 <= input_shape[-2] else (1,1) # Default value
        else:
            kernel_shape = node_attrs['kernel_shape']
            if isinstance(kernel_shape, int):
                kernel_shape = (kernel_shape, kernel_shape)
            if len(kernel_shape) == 1:
                kernel_shape = (kernel_shape[0], kernel_shape[0])
            if len(kernel_shape) > 2:      
                kernel_shape = kernel_shape[-2:] # Take the last two values
            if len(kernel_shape) == 0:
                kernel_shape = (3, 3) if input_shape[-1] >= 3 <= input_shape[-2] else (1,1) # Default value

        # Make sure dialations is matching kernel_shape
        if 'dialations' not in node_attrs:
            dilations = (1, 1)
        else:
            dilations = node_attrs['dilations']
            if isinstance(dilations, int):
                dilations = (dilations, dilations)
            if len(dilations) == 1:
                dilations = (dilations[0], dilations[0])
            if len(dilations) > 2:      
                dilations = dilations[-2:]
            if len(dilations) == 0:
                dilations = (1, 1)


        # Make sure pads is matching kernel_shape
        if 'pads' not in node_attrs:
            pads = (0, 0, 0, 0)
        else:
            pads = node_attrs['pads']
            if isinstance(pads, int):
                pads = (pads, pads, pads, pads)
            if len(pads) == 1:
                pads = (pads[0], pads[0], pads[0], pads[0])
            if len(pads) == 2:
                pads = (pads[0], pads[1], pads[0], pads[1])
            if len(pads) > 4:
                pads = pads[-4:] # Take the last four values
            if len(pads) == 0:
                pads = (0, 0, 0, 0)
        
        
        # Make sure strides is matching kernel_shape
        if 'strides' not in node_attrs:
            strides = (1, 1)
        else:
            strides = node_attrs['strides']
            if isinstance(strides, int):
                strides = (strides, strides)
            if len(strides) == 1:
                strides = (strides[0], strides[0])
            if len(strides) > 2:
                strides = strides[-2:]
            if len(strides) == 0:
                strides = (1, 1)

        if 'group' not in node_attrs:
            group = 1
        else:
            group = node_attrs['group']
            in_channels = input_shape[1]
            out_channels = round(input_shape[1]/2) # halfing the number of channels
            if out_channels == 0:
                out_channels = 1
            if (out_channels < group) or (in_channels % group != 0 or out_channels % group != 0):
                group = 1

        node_attrs['kernel_shape'] = kernel_shape
        node_attrs['strides'] = strides
        node_attrs['dilations'] = dilations
        node_attrs['pads'] = pads
        node_attrs['group'] = group

        # Fallback to default values if kernel_shape or strides are larger than input_shape
        if kernel_shape[0] > input_shape[-2] or kernel_shape[1] > input_shape[-1]:
            kernel_shape = (1, 1) # Default value
        if strides[0] > input_shape[-2] or strides[1] > input_shape[-1]:
            strides = (1, 1)
        
        weight_shape = [out_channels, in_channels // group, *kernel_shape]

        # Create weight tensor
        input_name = f"Conv_{node}"
        weights = np.random.rand(*weight_shape).astype(np.float32)
        input = helper.make_tensor(
            name=input_name,
            data_type=onnx.TensorProto.FLOAT,
            dims=weight_shape,
            vals=weights.flatten(),
        )
    elif attrs['operator']['op_type'] == 'MatMul':
        output_dim = 2 # choose yourself
        y_values = np.random.rand(input_shape, output_dim).astype(np.float32)
        # Create the tensor for Y
        input_name = f"Matmul_{node}"
        input = helper.make_tensor(
            name=input_name,
            data_type=onnx.TensorProto.FLOAT,
            dims=(input_shape, output_dim),
            vals=y_values.flatten()
        )
    elif attrs['operator']['op_type'] == 'Add':
        add_dim = input_shape
        add_values = np.random.rand(*add_dim).astype(np.float32)
        # Create the tensor for Y
        input_name = f"Add_{node}"
        input = helper.make_tensor(
            name=input_name,
            data_type=onnx.TensorProto.FLOAT,
            dims=add_dim,
            vals=add_values.flatten()
        )
    elif attrs['operator']['op_type'] == 'Sub':
        sub_dim = input_shape
        sub_values = np.random.rand(*sub_dim).astype(np.float32)
        # Create the tensor for Y
        input_name = f"Sub_{node}"
        input = helper.make_tensor(
            name=input_name,
            data_type=onnx.TensorProto.FLOAT,
            dims=sub_dim,
            vals=sub_values.flatten()
        )
    elif attrs['operator']['op_type'] == 'Div':
        div_dim = input_shape
        div_values = np.random.rand(*div_dim).astype(np.float32)
        # Create the tensor for Y
        input_name = f"Div_{node}"
        input = helper.make_tensor(
            name=input_name,
            data_type=onnx.TensorProto.FLOAT,
            dims=div_dim,
            vals=div_values.flatten()
        )
    elif attrs['operator']['op_type'] == 'Mul':
        mul_dim = input_shape
        mul_values = np.random.rand(*mul_dim).astype(np.float32)
        # Create the tensor for Y
        input_name = f"Mul_{node}"
        input = helper.make_tensor(
            name=input_name,
            data_type=onnx.TensorProto.FLOAT,
            dims=mul_dim,
            vals=mul_values.flatten()
        )
    elif attrs['operator']['op_type'] == 'Gemm':
        # Extract input dimensions for A from the shape tracker
        a_shape = input_shape

        if len(a_shape) == 3:  # Batch dimension present
            batch_size, m, k = a_shape
        elif len(a_shape) == 4:
            batch_size, m, k, _ = a_shape
        else:
            batch_size = None
            m, k = a_shape

        # Handle transA
        transA = node_attrs.get('transA', 0)
        if transA == 1:
            m, k = k, m

        # Define dimensions for B (with transposition)
        n = 4  # Default to N = 4; adjust as needed
        transB = node_attrs.get('transB', 0)
        if transB == 0:
            b_shape = (k, n)  # No transpose for B
        else:
            b_shape = (n, k)  # Transpose B, so B has shape (N, K)

        # Create B tensor
        b_values = np.random.rand(*b_shape).astype(np.float32)
        b_input_name = f"Gemm_B_{node}"
        b_input = helper.make_tensor(
            name=b_input_name,
            data_type=onnx.TensorProto.FLOAT,
            dims=b_shape,
            vals=b_values.flatten()
        )

        # Handle C (bias term) and broadcasting
        c_shape = (n,)  # Bias should be of shape (N,) for broadcasting
        c_values = np.random.rand(*c_shape).astype(np.float32)
        c_input_name = f"Gemm_C_{node}"
        c_input = helper.make_tensor(
            name=c_input_name,
            data_type=onnx.TensorProto.FLOAT,
            dims=c_shape,
            vals=c_values.flatten()
        )

        # Update the shape of the output
        output_shape = (batch_size, m, n) if batch_size is not None else (m, n)
        shape_tracker[node] = output_shape

        # Return B and C as the inputs for the Gemm node
        input_name = [b_input_name, c_input_name]
        input = [b_input, c_input]
            
    return input, input_name

def _prepare_for_output(onnx_nodes, initializers, shape_tracker, output_shape):
    second_to_last_shape = list(shape_tracker.values())[-1]
    second_to_last_shape_name = list(shape_tracker.keys())[-1]
    if second_to_last_shape != output_shape:
        # Create the Reshape node
        flatten_output_name = f"custom_flatten_output_{len(onnx_nodes)}"
        flatten_node = helper.make_node(
            'Flatten',
            name=f"custom_flatten_{len(onnx_nodes)}",
            inputs=[second_to_last_shape_name],
            outputs=[flatten_output_name]
        )

        flattened_shape = np.prod(second_to_last_shape[1:])

        # Create a fully connected node to match the output shape
        fully_weights = np.random.rand(flattened_shape, output_shape[1]).astype(np.float32)
        fully_bias = np.random.rand(output_shape[1]).astype(np.float32)
        fully_weights_name = f"custom_weights"
        fully_bias_name = f"custom_bias"
        fully_weights_tensor = helper.make_tensor(
            name=fully_weights_name,
            data_type=onnx.TensorProto.FLOAT,
            dims=fully_weights.shape,
            vals=fully_weights.flatten()
        )
        fully_bias_tensor = helper.make_tensor(
            name=fully_bias_name,
            data_type=onnx.TensorProto.FLOAT,
            dims=fully_bias.shape,
            vals=fully_bias.flatten()
        )

        initializers.append(fully_weights_tensor)
        initializers.append(fully_bias_tensor)

        fully_connected_output_name = f"custom_fully_connected_output_{len(onnx_nodes)}"
        fully_connected_node = helper.make_node(
            'Gemm',
            name=f"custom_fully_connected_{len(onnx_nodes)}",
            inputs=[flatten_output_name, fully_weights_name, fully_bias_name],
            outputs=[fully_connected_output_name],
            transA = 0,
            transB = 0,
            alpha = 1.0,
            beta = 1.0
        )

        output_node = onnx_nodes.pop()
        output_node.input[0] = fully_connected_output_name

        # If last node is Softmax, LogSoftmax then change axis to 1
        if output_node.op_type in ['Softmax', 'LogSoftmax']:
            for attr in output_node.attribute:
                if attr.name == 'axis':
                    attr.i =0 
                    break

        onnx_nodes.append(flatten_node)
        onnx_nodes.append(fully_connected_node)
        onnx_nodes.append(output_node)
        
        # Update the shape tracker with the new shape
        shape_tracker[flatten_output_name] = (1, flattened_shape)
        shape_tracker[fully_connected_output_name] = output_shape


def networkx_to_onnx(nx_graph, input_shape, output_shape):
    nodes = list(nx_graph.nodes(data=True))
    edges = list(nx_graph.edges(data=True))

    # Create ONNX nodes
    onnx_nodes = []
    initializers = []
    shape_tracker = {"input": input_shape}  # Initialize with input shape

    for node, attrs in nodes:
        # node_attrs = {}
        node_attrs = {k: v['value'] for k,v in attrs['attributes'].items()}

        # Define inputs for this node based on incoming edges
        input_names = [u for u, v, _ in edges if v == node]
        output_names = [v for u, v, _ in edges if u == node]
        output_nodes = [nx_graph.nodes[output_name] for output_name in output_names]
        if input_names == []:
            input_names = ["input"]

        for att, (att_type, val) in attrs['features']['attributes'].items():
            if att not in node_attrs:
                if val == 'NOTSET':
                    node_attrs[att] = val
                elif ast.literal_eval(val) is not None:
                    node_attrs[att] = __convert_attribute_to_type(val, att_type)

        pre_shape_op_input, pre_shape_op_input_name = pre_shape_inference_op_inputs(node, attrs, node_attrs, input_shape, output_nodes)
        if pre_shape_op_input is not None:
            if isinstance(pre_shape_op_input_name, list):
                input_names.extend(pre_shape_op_input_name)
                initializers.extend(pre_shape_op_input)
            else:
                input_names.append(pre_shape_op_input_name)
                initializers.append(pre_shape_op_input)
        
        # Infer shapes with a temporary ONNX model
        onnx_node = helper.make_node(
            attrs['operator']['op_type'],
            inputs=input_names,
            outputs=[node],
            name=node,
            **node_attrs,
        )
        onnx_nodes.append(onnx_node)
        temp_graph = helper.make_graph(
            onnx_nodes,
            "graph",
            inputs=[helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, input_shape)],
            outputs=[helper.make_tensor_value_info(onnx_nodes[-1].name, onnx.TensorProto.FLOAT, None)],
            initializer=initializers
        )
        temp_model = helper.make_model(temp_graph)
        inferred_model = infer_shapes(temp_model, data_prop=True)
        
        # Update the shape tracker with inferred shapes
        inferred_shapes = {vi.name: [dim.dim_value for dim in vi.type.tensor_type.shape.dim] for vi in inferred_model.graph.value_info}
        shape_tracker.update(inferred_shapes)

        post_shape_op_input, post_shape_op_input_name = post_shape_inference_op_inputs(node, attrs, node_attrs, shape_tracker)

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
            outputs=[node],
            name=node,
            **node_attrs,
        )
        onnx_nodes[-1] = onnx_node
    
    _prepare_for_output(onnx_nodes, initializers, shape_tracker, output_shape)
    # Define inputs and outputs
    inputs = [helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, input_shape)]
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
    return onnx_model

def onnx_to_pytorch(onnx_model: Union[ModelProto, str]) -> object:
    if isinstance(onnx_model, str):
        onnx_model = onnx.load(onnx_model)
    
    shaped_onnx_model = infer_shapes(onnx_model)

    check_model(shaped_onnx_model)

    return ONNX2PytorchConvert(onnx_model)


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

def tf_to_networkx(tf_model: tf_keras.Model, hash_nodes:bool=False, with_data:bool=False) -> nx.DiGraph:
    concrete_func = tf_model.signatures['serving_default']
    tf_graph =concrete_func.graph.as_graph_def()
    function_library = tf_graph.library
    function_defs = [f for f in function_library.function]

    graph_nodes = [n for n in tf_graph.node]
    func_def_nodes = [n for n in function_defs[0].node_def]

    node_def_attrs = []
    for g_n, f_n in zip(graph_nodes, func_def_nodes):
        node_dict = MessageToDict(g_n)
        attributes = node_dict.get('attr', {})
        node_def_attrs.append({
            'op': f_n.op,
            'input': f_n.input,
            'output': f_n.name,
            'attr': attributes
        })

    # Create a NetworkX graph
    nx_graph = nx.DiGraph()
    for node in node_def_attrs:
        if hash_nodes:
            node_name = hash_node({'op': node['op'], 'attr': node['attr']})
        else:
            node_name = node['output']
        if with_data:
            nx_graph.add_node(node_name, operator=node['op'], attributes=node['attr'])
        else:
            nx_graph.add_node(node_name)

    return nx_graph