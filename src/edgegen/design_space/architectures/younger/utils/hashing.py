import os
from pathlib import Path
import json
import hashlib

def get_graph_paths(data_path: Path):
    # find all paths for network folders DATA_PATH/gaphd_id/network/
    graph_folders = []
    for root, dirs, files in os.walk(data_path):
        if 'network' in dirs:
            graph_folders.append(os.path.join(root, 'network'))
    return graph_folders
    
def make_json_serializable(obj):
    """Recursively convert non-serializable types to serializable ones."""
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")  # Convert bytes to UTF-8 string, replacing invalid characters
    elif isinstance(obj, set):
        return list(obj)  # Convert sets to lists
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}  # Recurse for dict
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]  # Recurse for list
    elif isinstance(obj, tuple):
        return tuple(make_json_serializable(item) for item in obj)  # Recurse for tuple
    else:
        return obj  # Return as-is if no special handling is required
    
def hash_node(node):
    """Hash a node to create a unique identifier for it."""
    serializable_node = make_json_serializable(node)
    node_str = json.dumps(serializable_node, sort_keys=True)
    return hashlib.sha256(node_str.encode()).hexdigest()