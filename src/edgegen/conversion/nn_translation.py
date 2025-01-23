import ast
import onnx
import random
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

def pre_shape_inference_op_inputs(node, attrs, node_attrs, shape_tracker, output_nodes):
    input = None
    input_name = None

    if attrs['operator']['op_type'] == 'Add':
        # if 'axis' in node_attrs remove it. Onnx Add does not have axis attribute
        if 'axis' in node_attrs:
            del node_attrs['axis']
        add_dim = list(shape_tracker.values())[-1]
        add_values = np.random.rand(*add_dim).astype(np.float32)
        # Create the tensor for Y
        input_name = f"Add_{node}"
        input = helper.make_tensor(
            name=input_name,
            data_type=onnx.TensorProto.FLOAT,
            dims=add_dim,
            vals=add_values.flatten()
        )
    elif attrs['operator']['op_type'] == 'Mul':
        mul_dim = list(shape_tracker.values())[-1]
        mul_values = np.random.rand(*mul_dim).astype(np.float32)
        # Create the tensor for Y
        input_name = f"Mul_{node}"
        input = helper.make_tensor(
            name=input_name,
            data_type=onnx.TensorProto.FLOAT,
            dims=mul_dim,
            vals=mul_values.flatten()
        )
    elif attrs['operator']['op_type'] == 'Reshape':
        reshape_dim = node_attrs.get('shape', [])
        input_name = f"Reshape_{node}"
        input = helper.make_tensor(
            name=input_name,
            data_type=onnx.TensorProto.INT64,
            dims=(len(reshape_dim),),
            vals=np.array(reshape_dim, dtype=np.int64)
        )
    elif attrs['operator']['op_type'] == 'Unsqueeze':
        axes = node_attrs.get('axes', [])
        input_name = f"Unsqueeze_{node}"
        input = helper.make_tensor(
            name=input_name,
            data_type=onnx.TensorProto.INT64,
            dims=(len(axes),),
            vals=np.array(axes, dtype=np.int64)
        )
    elif attrs['operator']['op_type'] == 'Concat':
        concat_dim = node_attrs.get('axis', 0)
        input_name = f"Concat_{node}"
        input = helper.make_tensor(
            name=input_name,
            data_type=onnx.TensorProto.INT64,
            dims=(1,),
            vals=np.array([concat_dim], dtype=np.int64)
        )

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

        group = 1
        in_channels = input_shape[1]
        out_channels = round(input_shape[1]/2) # halfing the number of channels
        if out_channels == 0:
            out_channels = 1

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
        output_dim = random.choice([8, 16, 32, 64, 128, 256, 512]) # choose yourself
        y_values = np.random.rand(*input_shape[1:], output_dim).astype(np.float32)
        # Create the tensor for Y
        input_name = f"Matmul_{node}"
        input = helper.make_tensor(
            name=input_name,
            data_type=onnx.TensorProto.FLOAT,
            dims=(input_shape, output_dim),
            vals=y_values.flatten()
        )
    elif attrs['operator']['op_type'] == 'Sub':
        sub_dim = input_shape[:]
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
        div_dim = input_shape[1:]
        div_values = np.random.rand(*div_dim).astype(np.float32)
        # Create the tensor for Y
        input_name = f"Div_{node}"
        input = helper.make_tensor(
            name=input_name,
            data_type=onnx.TensorProto.FLOAT,
            dims=div_dim,
            vals=div_values.flatten()
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
    output_node = onnx_nodes.pop()
    if second_to_last_shape != output_shape:
        # Create the Reshape node
        flatten_output_name = f"custom_final_flatten"
        flatten_node = helper.make_node(
            'Flatten',
            name=f"custom_final_flatten",
            inputs=[second_to_last_shape_name],
            outputs=[flatten_output_name],
            axis=1
        )

        flattened_shape = int(np.prod(second_to_last_shape[1:]))
        onnx_nodes.append(flatten_node)
        shape_tracker[flatten_output_name] = (1, flattened_shape)

        W = np.random.randn(flattened_shape, output_shape[1]).astype(np.float32)
        W_name = f"custom_final_matmul_W"
        W_initializer = helper.make_tensor(
            name=W_name,
            data_type=onnx.TensorProto.FLOAT,
            dims=W.shape,
            vals=W.flatten().tolist()
        )
        initializers.append(W_initializer)

        # Define the MatMul node
        matmul_output_name = f"custom_final_matmul"
        matmul_node = helper.make_node(
            'MatMul',
            name=f"custom_final_matmul",
            inputs=[flatten_output_name, W_name],
            outputs=[matmul_output_name]
        )
        onnx_nodes.append(matmul_node)
        shape_tracker[matmul_output_name] = output_shape

        # Optionally, define the bias vector B and Add node
        B = np.random.randn(output_shape[1]).astype(np.float32)
        B_name = f"custom_final_add_B"
        B_initializer = helper.make_tensor(
            name=B_name,
            data_type=onnx.TensorProto.FLOAT,
            dims=B.shape,
            vals=B.flatten().tolist()
        )
        initializers.append(B_initializer)

        add_output_name = f"custom_final_add"
        add_node = helper.make_node(
            'Add',
            name=f"custom_final_add",
            inputs=[matmul_output_name, B_name],
            outputs=[add_output_name]
        )
        onnx_nodes.append(add_node)
        shape_tracker[add_output_name] = output_shape

        output_node.input[0] = add_output_name

        # If last node is Softmax, LogSoftmax then change axis to 1
        if output_node.op_type in ['Softmax', 'LogSoftmax']:
            for attr in output_node.attribute:
                if attr.name == 'axis':
                    attr.i = 1 
                    break

        onnx_nodes.append(output_node)


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

        pre_shape_op_input, pre_shape_op_input_name = pre_shape_inference_op_inputs(node, attrs, node_attrs, shape_tracker, output_nodes)
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
        try:
            inferred_model = infer_shapes(temp_model, data_prop=True)
        except Exception as e:
            print(f"Error inferring shapes for node {node}: {e}")
        
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