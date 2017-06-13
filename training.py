import os
import numpy as np
from models import unet

from utils import save_model, segmentation_stats
import data_preprocessing

# constants
im_size = (512, 512, 1)
n_classes = 3
dataset_name = os.path.join('.', 'data', 'liver_dataset.npz')

# preprocess data if necessary
if not os.path.isfile(dataset_name):
    data_preprocessing.preprocess_data_folder(os.path.join('.', 'data'))

# load data
dataset = np.load(dataset_name)
x, y, xv, yv, m, s = dataset['x'], dataset['y'], dataset['xv'], dataset['yv'], dataset['m'], dataset['s']

model = unet(
    im_size=im_size, output_size=n_classes,
    n_blocks=6, n_convs=1, n_filters=8
)

# training parameters
bs = 4
epochs = 6

# train
model.m = m
model.s = s
history = model.fit(x, y, batch_size=bs, epochs=epochs, validation_data=(xv, yv), shuffle=True)

# save results
save_model(model, history, 'livernet')

# evaluate
y_hat = model.predict(xv)

# prepare data for the two wanted tasks
true_liver = yv[:, :2]
true_liver[:, 1] += yv[:, 2]
predicted_liver = y_hat[:, :2]
predicted_liver[:, 1] += y_hat[:, 2]

true_cancer = yv[:, [0, 2]]
predicted_cancer = y_hat[:, [0, 2]]

# collect statistics
dice_liver, ppv_liver, sens_liver = segmentation_stats(true_liver, predicted_liver, 2)
dice_ca, ppv_ca, sens_ca = segmentation_stats(true_cancer, predicted_cancer, 2)

print 'liver segmentation stats:'
print 'dice coeff = {}   ppv = {}   sensitivity = {}'.format(dice_liver, ppv_liver, sens_liver)
print 'cancer segmentation stats:'
print 'dice coeff = {}   ppv = {}   sensitivity = {}'.format(dice_ca, ppv_ca, sens_ca)
