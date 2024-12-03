# import tensorflow as tf
# from torch import nn
# import functools


# def convert(pt_model:nn.Module, resolution, tflite_fname):
#     # 1. convert the state_dict to tensorflow format
#     pt_sd = pt_model.state_dict()

#     tf_sd = {}
#     for key, v in pt_sd.items():
#         if key.endswith('depth_conv.conv.weight'):
#             v = v.permute(2, 3, 0, 1)
#         elif key.endswith('conv.weight'):
#             v = v.permute(2, 3, 1, 0)
#         elif key == 'classifier.linear.weight':
#             v = v.permute(1, 0)
#         tf_sd[key.replace('.', '/')] = v.numpy()

#     # 2. build the tf network using the same config
#     weight_decay = 0.

#     with tf.Graph().as_default() as graph:
#         with tf.Session() as sess:
#             def network_map(images):
#                 net_config = pt_model.config
#                 from .tf_modules import ProxylessNASNets
#                 net_tf = ProxylessNASNets(net_config=net_config, net_weights=tf_sd,
#                                           n_classes=pt_model.classifier.linear.out_features,
#                                           graph=graph, sess=sess, is_training=False,
#                                           images=images, img_size=resolution)
#                 logits = net_tf.logits
#                 return logits, {}

#             def arg_scopes_map(weight_decay=0.):
#                 arg_scope = tf.contrib.framework.arg_scope
#                 with arg_scope([]) as sc:
#                     return sc

#             slim = tf.contrib.slim

#             @functools.wraps(network_map)
#             def network_fn(images):
#                 arg_scope = arg_scopes_map(weight_decay=weight_decay)
#                 with slim.arg_scope(arg_scope):
#                     return network_map(images)

#             input_shape = [1, resolution, resolution, 3]
#             placeholder = tf.placeholder(name='input', dtype=tf.float32, shape=input_shape)

#             out, _ = network_fn(placeholder)

#             # 3. convert to tflite (with int8 quantization)
#             converter = tf.lite.TFLiteConverter.from_session(sess, [placeholder], [out])
#             converter.optimizations = [tf.lite.Optimize.DEFAULT]
#             converter.inference_output_type = tf.int8
#             converter.inference_input_type = tf.int8

#             converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

#             tflite_buffer = converter.convert()
#             tf.gfile.GFile(tflite_fname, "wb").write(tflite_buffer)


import torch
import tensorflow as tf
import numpy as np
import onnx
import onnx2tf
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Tuple

def convert(pt_model: torch.nn.Module, input_size: Tuple[int, int, int],
            model_path: Path,  
            n_calibrate_sample: int = 500):
    # 0. Create a calibration dataset loader
    calib_loader = DataLoader(torch.randn(n_calibrate_sample, *input_size[1:]), batch_size=1)

    # 1. Export the PyTorch model to ONNX format
    pt_model.eval()
    dummy_input = torch.randn(*input_size)
    onnx_path = model_path.with_suffix(".onnx")
    torch.onnx.export(pt_model, dummy_input, onnx_path, opset_version=13, input_names=['input'], output_names=['output'])

    # 3. Convert ONNX model to TensorFlow format
    tf_model_path = model_path.with_name(model_path.name + "_tf")
    onnx2tf.convert(input_onnx_file_path=onnx_path,  
                           output_folder_path=tf_model_path,
                           non_verbose=True,
                           not_use_onnxsim=True)

    # 4. Convert the TensorFlow model to TFLite with INT8 quantization
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path.as_posix())
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Define a representative dataset generator for calibration
    def representative_dataset_gen():
        for i, data in enumerate(calib_loader):
            if i >= n_calibrate_sample:
                break
            data = data.numpy().transpose(0, 2, 3, 1).astype(np.float32)
            yield [data]

    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    # Save the TFLite model
    with open(model_path.with_name(model_path.name + "_int8").with_suffix(".tflite"), "wb") as f:
        f.write(tflite_model)
