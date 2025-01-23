import os
from pathlib import Path
import json
import hashlib
import networkx as nx

def get_graph_paths(data_path: Path):
    # find all paths for network folders DATA_PATH/gaphd_id/network/
    graph_folders = []
    for root, dirs, files in os.walk(data_path):
        if 'network' in dirs:
            graph_folders.append(os.path.join(root, 'network'))
    return graph_folders
    
def make_json_serializable(obj):
    """Convert to JSON serializable format."""
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")  # Convert bytes to UTF-8 string, replacing invalid characters
    if isinstance(obj, str) and obj.startswith("b'") and obj.endswith("'"):
        return obj[2:-1]  # Remove b'' prefix and suffix from string
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}  # Recurse for dict
    elif isinstance(obj, (tuple, list, set)):
        return [make_json_serializable(item) for item in obj]  # Recurse for list
    else:
        return obj  # Return as-is if no special handling is required
    
def hash_node(node):
    """Hash a node to create a unique identifier for it."""
    node_str = json.dumps(node, sort_keys=True)
    return hashlib.sha256(node_str.encode()).hexdigest()

def create_graph_fingerprint(graph:nx.DiGraph, hash_nodes=False):
    """
    Hash a graph to create a unique identifier for it.
    The graph is expected to already have nodes where their names are hashes of the node data.
    """
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Graph must be a directed acyclic graph.")
    
    if hash_nodes:
        # Hash the nodes
        for node, attrs in graph.nodes(data=True):
            graph.nodes[node] = hash_node(attrs)

    ordered_nodes = list(nx.topological_sort(graph))
    return "-".join(ordered_nodes)