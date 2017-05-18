import os
import json
import numpy as np
from keras.models import model_from_json

def save_model(model, history, name):
    h = json.dumps(history.history)
    with open(name + '_history.txt', 'w') as f:
        f.write(h)
    model.save_weights(name + '_weights.h5')
    arch_str = model.to_json()
    with open(name + '_arch.txt', 'w') as f:
        f.write(arch_str)
    np.savez(name + '_normalization.npz', m=model.m, s=model.s)


def load_model(name):
    weight_name = name + '_weights.h5'
    arch_name = name + '_arch.txt'
    norm_name = name + '_normalization.npz'

    assert os.path.isfile(weight_name), 'weights file not found: {}'.format(weight_name)
    assert os.path.isfile(arch_name), 'arcitecture file not found: {}'.format(weight_name)
    assert os.path.isfile(norm_name), 'normalization file not found: {}'.format(norm_name)

    with open(arch_name, 'r') as f:
        arch_str = f.read()
    model = model_from_json(arch_str)
    model.load_weights(weight_name)

    norm_data = np.load(norm_name)
    model.m = norm_data['m']
    model.s = norm_data['s']

    return model
