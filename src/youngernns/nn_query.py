from pathlib import Path
from typing import Generator, List

from youngernns.utils.io import load_younger_network_paths
from youngernns.data.network import Network

def filter_networks_on_operators(data_path:Path, operators:List[str]) -> Generator[Network, None, None]:
    """
    Filter networks on operator type.

    Parameters
    ----------
    data_path : Path
        Path to the directory containing the networks.
    operator : str
        Operator to filter on.

    Returns
    -------
    Generator[Network, None, None]
        Generator of networks containing the operator.
    """
    for network_path in load_younger_network_paths(data_path):
        network = Network(data_path=network_path)
        
        if all([
            all([attr['operator']['op_type'] != operator for operator in operators]) 
            for _, attr in network.graph.nodes(data=True)
            ]):
            yield network

def select_networks_with_operators(data_path:Path, operators:List[str]) -> Generator[Network, None, None]:
    """
    Select networks based on a operator type.

    Parameters
    ----------
    data_path : Path
        Path to the directory containing the networks.
    operator : str
        Operator to filter on.

    Returns
    ------- 
    Generator[Network, None, None]
        Generator of networks containing the operator.
    """
    for network_path in load_younger_network_paths(data_path):
        network = Network(data_path=network_path)

        if any([
            any([attr['operator']['op_type'] == operator for operator in operators])
            for _, attr in network.graph.nodes(data=True)
            ]):
            yield network

if __name__ == "__main__":
    import os
    import argparse
    from tqdm import tqdm
    from youngernns.utils.logging import get_logger
    from youngernns.data.network import count_networks

    logger = get_logger(log_path_prefix=os.path.basename(__file__).replace(".py", ""))

    parser = argparse.ArgumentParser(description="Select networks on operator type.")
    parser.add_argument("--data_path", type=str, help="Path to the directory containing the networks.")
    parser.add_argument("--operators", type=str, nargs="+", help="Operators to select by.")

    args = parser.parse_args()

    logger.info(f"Selecting networks on operators: {args.operators}")
    try:
        data_path = Path(args.data_path)
        operators = args.operators

        network_count = count_networks(data_path)

        selected_count = 0
        for network in tqdm(select_networks_with_operators(data_path, operators), total=network_count):
            logger.info(network.data_path)
            selected_count += 1
        
        logger.info(f"Selected {selected_count} networks on operators: {operators}")
        print(f"Selected {selected_count} networks on operators: {operators}")
    except Exception as e:
        logger.error(f"Failed to select networks on operators: {e}")

