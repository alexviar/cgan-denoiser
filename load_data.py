import gzip
import os
#from six.moves.urllib.request import urlretrieve
import numpy
from tensorflow.keras import utils

def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def load_images(filename):
    '''Extract the images into a 4D uint8 numpy array [index, y, x, depth].'''
    print('Extracting', filename)

    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)

        if magic != 2051:
            raise ValueError(
                'Invalid magic number %d in MNIST image file: %s' %
                (magic, filename))

        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)

        buf = bytestream.read(rows * cols * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8)
        data = data.reshape(num_images, rows, cols)

        return data


def load_labels(filename):
    '''Extract the labels into a 1D uint8 numpy array [index].'''
    print('Extracting', filename)

    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)

        if magic != 2049:
            raise ValueError(
                'Invalid magic number %d in MNIST label file: %s' %
                (magic, filename))

        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8)

        return labels

def load(train_images_file, train_labels_file, test_images_file, test_labels_file):
  train_images = load_images(train_images_file)
  train_labels = load_labels(train_labels_file)
  test_images = load_images(test_images_file)
  test_labels = load_labels(test_labels_file)

  return ((train_images, train_labels),(test_images, test_labels))
