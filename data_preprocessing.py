import os
import numpy as np
from scipy.misc import imread
from glob import glob
from tqdm import tqdm

from utils import to_categorical


def preprocess_data_folder(data_folder):
    image_data_folder = os.path.join(data_folder, 'ct')
    label_folder = os.path.join(data_folder, 'seg')

    dataset = load_data(image_data_folder, label_folder,
                        im_size=(512, 512, 1), n_classes=3)

    dataset_normalized = normalize_data(dataset)

    dataset_file = os.path.join('.', 'data', 'liver_dataset.npz')
    np.savez(dataset_file, **dataset_normalized)


def load_data(data_folder, seg_folder,
              im_size=(128, 128, 3), n_classes=2, ext='png'):
    """loads images into numpy arrays"""
    train_list = glob(os.path.join(data_folder, 'train', '*.' + ext))
    val_list = glob(os.path.join(data_folder, 'val', '*.' + ext))

    n_train, n_val = len(train_list), len(val_list)

    dataset = init_dataset(im_size, n_train, n_val, n_classes)

    train_labels_folder = os.path.join(seg_folder, 'train')
    val_labels_folder = os.path.join(seg_folder, 'val')

    print 'packing training set'
    for train_i, im_name in enumerate(train_list):
        im, cat_label = read_image_and_label(
            im_name, train_labels_folder, im_size, n_classes)

        dataset['x'][train_i] = im
        dataset['y'][train_i] = cat_label

    print 'packing validation set'
    for val_i, im_name in enumerate(val_list):
        im, cat_label = read_image_and_label(
            im_name, val_labels_folder, im_size, n_classes)

        dataset['xv'][val_i] = im
        dataset['yv'][val_i] = cat_label

    return dataset


def init_dataset(im_size, n_train, n_val, n_classes):
    x = np.zeros((n_train, im_size[2], im_size[0], im_size[1]))
    y = np.zeros((n_train, n_classes, im_size[0], im_size[1]))
    xv = np.zeros((n_val, im_size[2], im_size[0], im_size[1]))
    yv = np.zeros((n_val, n_classes, im_size[0], im_size[1]))
    dataset = dict(x=x, y=y, xv=xv, yv=yv)
    return dataset


def read_image_and_label(im_name, labels_folder, im_size, n_classes):
    im = imread(im_name)
    im = adjust_color_channels(im, im_size)

    bname = os.path.basename(im_name)
    labelname = os.path.join(
        labels_folder, bname.replace('ct', 'seg'))

    cat_label = read_label(labelname, n_classes)

    im, cat_label = change_dim_ordering_to_theano(im, cat_label)

    return im, cat_label


def adjust_color_channels(im, im_size):
    if np.ndim(im) < len(im_size):
        im = np.expand_dims(im, 2)
    if im.shape[2] > im_size[2]:
        im = im[:, :, :im_size[2]]
    return im


def read_label(full_label_name, n_classes):
    scaled_label = imread(full_label_name)[:, :, 0] / 127
    cat_label = to_categorical(scaled_label, n_classes)
    return cat_label


def change_dim_ordering_to_theano(im, cat_label):
    im = np.rollaxis(im, 2)
    cat_label = np.rollaxis(cat_label, 2)
    return im, cat_label


def normalize_data(dataset):
    """for each sample in dataset subtracts pixelwise mean,
    and divides by pixelwise std"""

    n_train = dataset['x'].shape[0]
    n_val = dataset['xv'].shape[0]
    n = n_train + n_val

    mean_image = pixelwise_mean_memory_efficient(
        [dataset['x'], dataset['xv']], n)
    std_image = pixelwise_std_memory_efficient(
        [dataset['x'], dataset['xv']], n, mean_image)

    dataset['x'] = normalize_array_memory_efficient(
        dataset['x'], mean_image, std_image,
        msg='normalizing training data')

    dataset['xv'] = normalize_array_memory_efficient(
        dataset['xv'], mean_image, std_image,
        msg='normalizing validation data')

    dataset['m'] = mean_image
    dataset['s'] = std_image
    return dataset


def pixelwise_mean_memory_efficient(arrays, n_samples):
    mean_image = np.zeros(arrays[0][0].shape, dtype=np.float32)
    for arr in arrays:
        for sample in arr:
            mean_image += sample / float(n_samples)
    return mean_image


def pixelwise_std_memory_efficient(arrays, n_samples, mean_image):
    var_image = np.zeros(arrays[0][0].shape, dtype=np.float32)
    for arr in arrays:
        for sample in arr:
            var_image += (sample - mean_image) ** 2 / float(n_samples)
    std_image = np.sqrt(var_image)
    return std_image


def normalize_array_memory_efficient(x, mean_image, std_image, msg=''):
    for ind in tqdm(range(len(x)), desc=msg):
        x[ind] = (x[ind] - mean_image) / (std_image + 1e-12)
    return x
