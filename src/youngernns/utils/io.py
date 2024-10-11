# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-04-06 20:34
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.

import os
import toml
import json
import pickle
import psutil
import pathlib

def load_json(filepath: pathlib.Path | str) -> object:
    with open(filepath, 'r') as file:
        serializable_object = json.load(file)

    return serializable_object


def load_pickle(filepath: pathlib.Path | str) -> object:
    with open(filepath, 'rb') as file:
        data = pickle.load(file)

    serializable_object = pickle.loads(data['main'])

    return serializable_object