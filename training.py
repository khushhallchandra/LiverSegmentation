import os
import numpy as np
import json
from models import unet

# constants
im_size = (512, 512, 1)
n_classes = 3


# load data
dataset = np.load(os.path.join('.', 'data', 'liver_dataset.npz'))
x, y, xv, yv, m, s = dataset['x'], dataset['y'], dataset['xv'], dataset['yv'], dataset['m'], dataset['s']

model = unet(
    im_size=im_size, output_size=n_classes,
    n_blocks=6, n_convs=1, n_filters=8
)

# training parameters
bs = 4
epochs = 10

# train
history = model.fit(x,y, batch_size=bs, epochs=epochs,validation_data=(xv, yv), shuffle=True)

# save results
h = json.dumps(history.history)
with open('training_history.txt', 'w') as f:
    f.write(h)
model.save('livernet.model')
