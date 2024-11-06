import os
import numpy as np
import tensorflow as tf
from PIL import Image

def image_to_tflite_tensor(input_path, output_folder, shape=(512, 512)):
    # Load and preprocess the image
    image = Image.open(input_path).convert('RGB')
    image = image.resize(shape)
    
    # Convert image to a numpy array and normalize to [0, 1]
    image_array = np.array(image) / 255.0

    # Add batch dimension to fit TFLite input requirements
    tensor_image = np.expand_dims(image_array, axis=0).astype(np.float32)

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Define output path
    output_path = os.path.join(output_folder, f"{os.path.basename(input_path).split('.')[0]}_{shape[0]}_tensor.npy")

    # Save the tensor as a .npy file
    np.save(output_path, tensor_image)
    print(f"Image saved as TFLite-compatible tensor at: {output_path}")

if __name__ == '__main__':
    # Example usage:
    image_name = 'corgi.jpg'
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')
    input_path = os.path.join(root, 'data', 'images', image_name)
    output_folder = os.path.join(root, 'output', 'tf_images')
    image_to_tflite_tensor(input_path, output_folder, shape=(128, 128))
