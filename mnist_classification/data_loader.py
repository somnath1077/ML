import gzip
import numpy as np
import os

IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10

dir = os.path.dirname(__file__)
data_dir = dir + '/data/'
train_data_filename = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
train_labels_filename = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
test_data_filename = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
test_labels_filename = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')


def extract_data(filename, num_images):
    """Extract the images into a 2D matrix [image index, y * x].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, IMAGE_SIZE * IMAGE_SIZE)
        return data


def extract_labels(filename, num_images):
    """Extract the labels into a vector of int64 label IDs."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels


def load_training_data():
    train_data = extract_data(train_data_filename, 60000)
    train_labels = extract_labels(train_labels_filename, 60000)
    return train_data, train_labels


def load_test_data():
    test_data = extract_data(test_data_filename, 10000)
    test_labels = extract_labels(test_labels_filename, 10000)
    return test_data, test_labels


if __name__ == '__main__':
    X, y = load_training_data()
    print("loaded training data")
