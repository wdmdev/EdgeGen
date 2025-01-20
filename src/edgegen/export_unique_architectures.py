import os
import shutil
import sys
from argparse import ArgumentParser
from tqdm import tqdm

def copy_files(source_dir, file_list_path):
    # Check if the file exists
    if not os.path.isfile(file_list_path):
        print(f"Error: File '{file_list_path}' not found!")
        sys.exit(1)

    # Read the file paths from the .txt file
    with open(file_list_path, 'r') as file:
        lines = file.readlines()

    for line in tqdm(lines):
        file_path = line.strip()  # Remove leading/trailing whitespace
        if os.path.isfile(file_path):
            try:
                print(f"Copying {file_path} and its tf dir to the current directory...")
                tf_dir = file_path.split("_int8.tflite")[0] + "_tf"
                shutil.copytree(tf_dir, os.path.join(source_dir, os.path.basename(tf_dir)))
                shutil.copy(file_path, source_dir)
            except Exception as e:
                print(f"Error copying {file_path}: {e}")
        else:
            print(f"Warning: {file_path} does not exist or is not a regular file.")

    print("Done!")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--source_dir", type=str, help="Path to the directory containing the files to copy",)
    parser.add_argument("--file_list_path", type=str, help="Path to the file containing the list of files to copy")
    args = parser.parse_args()

    # Get the input file path from command-line arguments
    copy_files(args.source_dir, args.file_list_path)
