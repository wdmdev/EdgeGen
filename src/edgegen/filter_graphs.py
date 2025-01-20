import os
from argparse import ArgumentParser
from pathlib import Path
from edgegen.conversion.nn_translation import tf_to_networkx
import tensorflow as tf
from edgegen.design_space.architectures.younger.utils.hashing import create_graph_fingerprint
import glob
from tqdm import tqdm
import uuid

# Setup logging and make logger log to output/log_filter_graphs_MMDDYYYY.log
import logging
from datetime import datetime
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_file = Path(__file__).parent.parent.parent / "output" / f"log_filter_graphs_{datetime.now().strftime('%m%d%Y')}.log"
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--folder_path", type=str, required=False)
    args = parser.parse_args()

    output_dir = Path(__file__).parent.parent.parent / "output" / f"graph_filtering_{uuid.uuid4().hex}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    unique_models = []
    duplicate_models = []
    duplicate_of = []
    empty_models = []
    paths_to_unique_models = []

    all_tflite_files = glob.glob(os.path.join(args.folder_path, '**', '*_int8.tflite'), recursive=True)

    for file_path in tqdm(all_tflite_files):
        if file_path.endswith("_int8.tflite"):
            tf_dir = file_path.split("_int8.tflite")[0] + "_tf" 
            if not os.path.exists(os.path.join(tf_dir, "saved_model.pb")):
                empty_models.append(file_path)
                continue
            else:
                # Load keras model
                tf_model = tf.saved_model.load(tf_dir)
                # Convert to onnx
                networkx_model = tf_to_networkx(tf_model, hash_nodes=True)
                graph_fingerprint = create_graph_fingerprint(networkx_model)

                if graph_fingerprint not in unique_models:
                    unique_models.append(graph_fingerprint)
                    paths_to_unique_models.append(file_path)    
                else:
                    duplicate_models.append(file_path)
                    duplicate_of.append(paths_to_unique_models[unique_models.index(graph_fingerprint)])
    
    # Log results
    logger.info(f"Folder path: {args.folder_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Number of total models: {len(all_tflite_files)}")
    logger.info(f"Number of unique models: {len(unique_models)}")
    logger.info(f"Number of duplicate models: {len(duplicate_models)}")
    logger.info(f"Number of empty models: {len(empty_models)}")

    # Save unique model paths
    with open(output_dir / "unique_models.txt", "w") as f:
        for path in paths_to_unique_models:
            f.write(f"{path}\n")
        
    # Save duplicate model paths as .csv
    with open(output_dir / "duplicate_models.csv", "w") as f:
        f.write("duplicate_model, duplicate_of\n")
        for i in range(len(duplicate_models)):
            f.write(f"{duplicate_models[i]}, {duplicate_of[i]}\n")
    
    # Save empty model paths
    with open(output_dir / "empty_models.txt", "w") as f:
        for path in empty_models:
            f.write(f"{path}\n")