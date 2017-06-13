"""utility functions for this project"""
import os
import json
import numpy as np
from keras.models import model_from_json


def to_categorical(im, n_classes):
    """converts label image to categorical numpy array"""
    s = im.shape
    categorical = np.zeros((s[0], s[1], n_classes), dtype=np.uint8)
    for class_i in range(n_classes):
        categorical[:, :, class_i] = im == class_i
    return categorical


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

    assert os.path.isfile(weight_name), \
        'weights file not found: {}'.format(weight_name)

    assert os.path.isfile(arch_name), \
        'arcitecture file not found: {}'.format(weight_name)

    assert os.path.isfile(norm_name), \
        'normalization file not found: {}'.format(norm_name)

    with open(arch_name, 'r') as f:
        arch_str = f.read()
    model = model_from_json(arch_str)
    model.load_weights(weight_name)

    norm_data = np.load(norm_name)
    model.m = norm_data['m']
    model.s = norm_data['s']

    return model


def segmentation_stats(y_true, y_pred, n_classes):
    labels = np.argmax(y_pred, axis=1)
    prediction = np.array(
        [np.rollaxis(to_categorical(label, n_classes=n_classes), 2)
         for label in labels])
    tp = np.count_nonzero(np.logical_and(prediction == 1, y_true == 1))
    fp = np.count_nonzero(np.logical_and(prediction == 1, y_true == 0))
    fn = np.count_nonzero(np.logical_and(prediction == 0, y_true == 1))
    tn = np.count_nonzero(np.logical_and(prediction == 0, y_true == 0))
    dice = 2.0 * tp / (2.0 * tp + fn + fp)
    ppv = tp / float(tp + fp)
    sens = tp / float(tp + fn)
    return dice, ppv, sens
