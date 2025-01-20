import argparse
import numpy as np
import tensorflow as tf
from PIL import Image

def load_and_predict(model_path: str, image_path: str):
    # Load the TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load and preprocess the image
    img = Image.open(image_path).convert("RGB")
    input_shape = input_details[0]['shape']
    img = img.resize((input_shape[2], input_shape[1]))
    input_data = np.array(img, dtype=np.float32) / 255.0

    # Quantize the input data
    input_scale, input_zero_point = input_details[0]['quantization']
    input_data = input_data / input_scale + input_zero_point
    input_data = input_data.astype(np.int8)
    input_data = np.expand_dims(input_data, axis=0)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Invoke the interpreter
    interpreter.invoke()

    # Retrieve and dequantize the output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_scale, output_zero_point = output_details[0]['quantization']
    output_data = output_scale * (output_data.astype(np.float32) - output_zero_point)

    # Interpret the output (example for classification)
    print(output_data.shape)
    predicted_class = np.argmax(output_data)
    print(predicted_class)
    confidence = output_data[0][predicted_class]

    print(f"Predicted class: {predicted_class} with confidence {confidence:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on an image using a TFLite model.")
    parser.add_argument("model_path", type=str, help="Path to the TFLite model file.")
    parser.add_argument("image_path", type=str, help="Path to the input image file.")
    args = parser.parse_args()

    load_and_predict(args.model_path, args.image_path)
