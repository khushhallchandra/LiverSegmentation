import os
import numpy as np
from scipy.misc import imread
from glob import glob
from tqdm import tqdm


def to_categorical(im, n_classes):
    """converts label image to categorical numpy array"""
    s = im.shape
    categorical = np.zeros((s[0], s[1], n_classes), dtype=np.uint8)
    for class_i in range(n_classes):
        categorical[:, :, class_i] = im == class_i
    return categorical


def load_data(data_folder, seg_folder, im_size=(128, 128, 3), n_classes=2, ext='png'):
    """loads images into numpy arrays"""
    train_list = glob(os.path.join(data_folder, 'train', '*.' + ext))
    val_list = glob(os.path.join(data_folder, 'val', '*.png'))

    n_train = len(train_list)
    n_val = len(val_list)

    # init numpy arrays
    x = np.zeros((n_train, im_size[2], im_size[0], im_size[1]))
    y = np.zeros((n_train, n_classes, im_size[0], im_size[1]))
    xv = np.zeros((n_val, im_size[2], im_size[0], im_size[1]))
    yv = np.zeros((n_val, n_classes, im_size[0], im_size[1]))

    for train_i, imname in enumerate(train_list):
        bname = os.path.basename(imname)
        labelname = os.path.join(seg_folder, 'train', bname.replace('ct', 'seg'))

        im = imread(imname)
        cat_label = to_categorical(imread(labelname)[:, :, 0] / 127, n_classes)

        if np.ndim(im) < len(im_size):
            im = np.expand_dims(im, 2)
        if im.shape[2] > im_size[2]:
            im = im[:, :, :im_size[2]]

        # convert to theano dim ordering
        im = np.rollaxis(im, 2)
        cat_label = np.rollaxis(cat_label, 2)

        x[train_i] = im
        y[train_i] = cat_label

        # sanity check
        # plt.figure(1)
        # plt.clf()
        # plt.imshow(im[:,:,0], cmap='gray')
        # plt.imshow(label[:,:,0], cmap='jet', alpha=0.3, vmin=0, vmax=2)
        # plt.pause(0.2)

    for val_i, imname in enumerate(val_list):
        bname = os.path.basename(imname)
        labelname = os.path.join(seg_folder, 'val', bname.replace('ct', 'seg'))

        im = imread(imname)
        cat_label = to_categorical(imread(labelname)[:, :, 0] / 127, n_classes)

        if np.ndim(im) < len(im_size):
            im = np.expand_dims(im, 2)
        if im.shape[2] > im_size[2]:
            im = im[:, :, :im_size[2]]

        # convert to theano dim ordering
        im = np.rollaxis(im, 2)
        cat_label = np.rollaxis(cat_label, 2)

        xv[val_i] = im
        yv[val_i] = cat_label
    return x, y, xv, yv


def normalize_data(x, xv):
    """for each sample in dataset subtracts pixelwise mean, and divides by pixelwise std"""
    m = np.zeros((x.shape[1], x.shape[2], x.shape[3]))
    s2 = np.zeros((x.shape[1], x.shape[2], x.shape[3]))
    s = np.zeros((x.shape[1], x.shape[2], x.shape[3]))

    n_train = x.shape[0]
    n_val = xv.shape[0]
    n = n_train + n_val

    # calculate pixelwise mean
    for sample in x:
        m += sample / float(n)
    for sample in xv:
        m += sample / float(n)

    # calculate pixelwise std
    for sample in x:
        s2 += (sample - m) ** 2 / float(n)
    for sample in xv:
        s2 += (sample - m) ** 2 / float(n)

    s = np.sqrt(s2)
    s += 1  # to avoid division by zero

    # noralize data
    for train_i in tqdm(range(x.shape[0]), desc='normalizing training data'):
        x[train_i] = (x[train_i] - m) / s
    for val_i in tqdm(range(xv.shape[0]), desc='normalizing val data'):
        xv[val_i] = (xv[val_i] - m) / s

    return x, xv, m, s


if __name__ == '__main__':
    # import matplotlib.pyplot as plt

    data_folder = os.path.join('.', 'data', 'ct')
    label_folder = os.path.join('.', 'data', 'seg')

    x, y, xv, yv = load_data(data_folder, label_folder, im_size=(512, 512, 1), n_classes=3)
    x, xv, m, s = normalize_data(x, xv)

    np.savez(os.path.join('.', 'data','liver_dataset.npz'), x=x, y=y, xv=xv, yv=yv, m=m, s=s)
