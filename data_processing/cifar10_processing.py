"""
Processing of the CIFAR-10 data, downloaded from https://www.cs.toronto.edu/~kriz/cifar.html

Reference:
https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf, Alex Krizhevsky
"""
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


data_path = "/home/kevin/deep_learning/cifar-10-python/cifar-10-batches-py/"

batches = []
for i in range(1,6):
    dict = unpickle(data_path + "data_batch_" + str(i))
    batches.append(np.asarray(dict[b'data']).astype("uint8"))

X = np.concatenate(batches)

x = np.asarray([np.reshape(b, [32, 32, 3]) for b in X])

data = tf.data.Dataset.from_tensor_slices(x)

data = data.shuffle(10).repeat().batch(batch_size=2)