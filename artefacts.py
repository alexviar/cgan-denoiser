# -*- coding: utf-8 -*-
"""
artefacts.py

Functions to artificially induce artefacts in images. 

author: Ben Cottier (git: bencottier)
"""
from __future__ import absolute_import, division, print_function
from data_processing import normalise
from PIL import Image, ImageDraw
import numpy as np
import random

getit = lambda : (random.randrange(0, 27),random.randrange(0, 27))

def add_noise(data):
    noisy = []
    for img in data:
        #img = img.transpose()
        #        img = img + random.random()*0.33*np.random.normal(0, 255, img.shape)
        pilimg = Image.fromarray(img)
        draw = ImageDraw.Draw(pilimg)
        # draw some random lines
        for i in range(5,random.randrange(6, 8)):
            draw.line((getit(), getit()), fill=255, width=random.randrange(1,3))

        # draw some random points
        for i in range(10,random.randrange(11, 20)):
            draw.point((getit(), getit(), getit(), getit(), getit(), getit(), getit(), getit(), getit(), getit()), fill=random.randrange(220,255))
        noisy.append(np.array(pilimg))
    return np.array(noisy)

def add_gaussian_noise(data, stdev=0.1, mean=0.0, data_range=(0, 1), clip=True):
    """
    Add noise to array data, sampled from a normal/Gaussian distribution.

    Assumes the data is limited to the range [-1.0, 1.0].

    Arguments:
        data: ndarray. Data to add noise to.
        stdev: float > 0. Standard deviation of the noise distribution.
        mean: float. Mean (average) of the noise distribution.
        clamping: bool. If True, limit the resulting data to [-1.0, 1.0]
    """
    data_ = normalise(data, (-1, 1), data_range)
    noisy = data_ + np.random.normal(mean, stdev, data.shape)
    noisy = np.clip(noisy, -1, 1) if clip else noisy
    return normalise(noisy, data_range, (-1, 1))


def clamp(x, rng=(-1, 1)):
    return np.maximum(rng[0], np.minimum(x, rng[1]))


if __name__ == "__main__":
    # Test
    from scipy.misc import ascent
    import tensorflow as tf
    import matplotlib.pyplot as plt

    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

    # data = ascent()
    data = train_images
    print("Data shape: ", data.shape)
    #data = add_gaussian_noise(normalise(data, (-1, 1), (0, 255)), 0.2)
    data = add_gaussian_noise(data, 0.2)
    data2 = add_noise(data)
    # plt.imshow(data[0], cmap='gray')

    fig = plt.figure(figsize=(8,4))

    for i in range(32):
        plt.subplot(8, 4, i+1)
        plt.imshow(data[i, :, :], cmap='gray')
        plt.imshow(data2[i, :, :], cmap='gray')
        plt.axis('off')

    plt.show()
