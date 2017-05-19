import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from scipy.misc import imread, imsave

from utils import load_model

# constants
model_name = os.path.join('trained_model', 'livernet')
im_size = (512, 512, 1)
display = True
save = False
ext = 'png'
data_folder = os.path.join('.', 'data', 'ct')
label_folder = os.path.join('.', 'data', 'seg')

# load model
model = load_model(model_name)

# read test set
test_list = glob(os.path.join(data_folder, 'test', '*.' + ext))
n_test = len(test_list)

# predict on test set
for fname in test_list:
    im = imread(fname)[:, :, 0]
    normalized_im = (im.astype(np.float) - model.m) / model.s
    normalized_im = np.expand_dims(normalized_im, 0)
    prediction = model.predict(normalized_im)

    labels = np.argmax(prediction, axis=1)[0, :, :]

    if display:
        plt.figure(1)
        plt.clf()
        plt.imshow(im, cmap='gray')
        plt.imshow(labels, cmap='jet', alpha=0.3, vmin=0, vmax=2)
        plt.pause(0.1)

    if save:
        bname = os.path.basename(fname)
        bname = bname.replace('ct', 'seg')
        l_name = os.path.join(label_folder, 'test', bname)
        imsave(l_name, labels)