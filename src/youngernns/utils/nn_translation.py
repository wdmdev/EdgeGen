
import ast
import onnx
from onnx import helper
from onnx.shape_inference import infer_shapes
import numpy as np

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

        if attrs['operator']['op_type'] == 'Reshape':
            batch_size = input_shape[0]
            target_shape = [batch_size, -1]
            shape_tensor_name = f"Reshape_shape_{node}"
            shape_tensor = helper.make_tensor(
                shape_tensor_name,
                onnx.TensorProto.INT64,
                dims=(len(target_shape),),
                vals=target_shape
            )
            initializers.append(shape_tensor)
            input_names.append(shape_tensor_name)
        
        onnx_node = helper.make_node(
            attrs['operator']['op_type'],
            inputs=input_names,
            outputs=[f"node_{node}"],
            name=f"node_{node}",
            **node_attrs,
        )
        onnx_nodes.append(onnx_node)

        # Infer shapes with a temporary ONVNX model
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

        if attrs['operator']['op_type'] == "Conv":
            num_filters = 3  # You may adjust this as needed
            kernel_shape = node_attrs['kernel_shape']
            group = node_attrs['group']
            num_channels = shape_tracker[input_names[0]][1]
            weight_shape = [num_filters, num_channels // group, *kernel_shape]

            # Create weight tensor
            weight_tensor_name = f"Conv_weights_{node}"
            weights = np.random.rand(*weight_shape).astype(np.float32)
            weight_tensor = helper.make_tensor(
                name=weight_tensor_name,
                data_type=onnx.TensorProto.FLOAT,
                dims=weight_shape,
                vals=weights.flatten(),
            )
            initializers.append(weight_tensor)
            # Insert the weight tensor as the second input
            input_names.append(weight_tensor_name)
        
        if attrs['operator']['op_type'] == 'MatMul':
            input_dim = shape_tracker[input_names[0]][1]
            output_dim = 2 # choose yourself
            y_values = np.random.rand(input_dim, output_dim).astype(np.float32)
            # Create the tensor for Y
            matmul_input_name = f"Matmul_Y_{node}"
            y_tensor = helper.make_tensor(
                name=matmul_input_name,
                data_type=onnx.TensorProto.FLOAT,
                dims=(input_dim, output_dim),
                vals=y_values.flatten().tolist()
            )
            initializers.append(y_tensor)
            input_names.append(matmul_input_name)
        
        if attrs['operator']['op_type'] == 'Add':
            add_dim = shape_tracker[input_names[0]]
            add_values = np.random.rand(*add_dim).astype(np.float32)
            # Create the tensor for Y
            add_input_name = f"Add_Y_{node}"
            add_tensor = helper.make_tensor(
                name=add_input_name,
                data_type=onnx.TensorProto.FLOAT,
                dims=add_dim,
                vals=add_values.flatten().tolist()
            )
            initializers.append(add_tensor)
            input_names.append(add_input_name)

        onnx_node = helper.make_node(
            attrs['operator']['op_type'],
            inputs=input_names,
            outputs=[f"node_{node}"],
            name=f"node_{node}",
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
