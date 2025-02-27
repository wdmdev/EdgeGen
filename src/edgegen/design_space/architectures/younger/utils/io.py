# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-04-06 20:34
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import pickle
import pathlib
from typing import Generator

def load_json(filepath: pathlib.Path | str) -> object:
    with open(filepath, 'r') as file:
        serializable_object = json.load(file)

    return serializable_object

def save_pickle(serializable_object: object, filepath: pathlib.Path | str) -> None:
    serialized_object = pickle.dumps(serializable_object)
    with open(filepath, 'wb') as file:
        pickle.dump(serialized_object, file)


def load_pickle(filepath: pathlib.Path | str) -> object:
    with open(filepath, 'rb') as file:
        data = pickle.load(file)

    serializable_object = pickle.loads(data['main'])

    return serializable_object

def load_younger_network_paths(data_path: pathlib.Path | str) -> Generator[str, None, None]:
    # find all paths for network folders DATA_PATH/gaphd_id/network/
    for root, dirs, files in os.walk(data_path):
        if 'network' in dirs:
            yield os.path.join(root, 'network')