import numpy as np
import matplotlib.pyplot as plt
import json
import os

def load_data(path):
    with open(path, "r") as file:
        data = json.load(file)
    return data

def load_flatten_weight(path):
    data = np.load(path)
    layer0 = data['arr_0'].reshape(data['arr_0'].shape[0], -1)
    layer1 = data['arr_1'].reshape(data['arr_1'].shape[0], -1)
    layer2 = data['arr_2'].reshape(data['arr_2'].shape[0], -1)
    flatten_weight = np.concatenate([layer0, layer1, layer2],axis=1)
    return flatten_weight

def load_multi_data(CASE, FOLDER, MODEL, TRIAL, BASE_PATH):
    data = []
    for trial in TRIAL:
        data_path = os.path.join(BASE_PATH, FOLDER, CASE, f'{MODEL}-pos_act_model0_buffer-{trial}.json')
        with open(data_path, "r") as file:
            data.append(json.load(file))
    return np.array(data)

