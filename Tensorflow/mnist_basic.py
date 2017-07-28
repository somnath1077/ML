import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# Placeholder variable to store an image (28 * 28 pixels = 784 pixels)
x = tf.placeholder(dtype=tf.float32, shape=[None, 784])

# Variables for the weight matrix and bias
W = tf.Variable(tf.zeros[784, 10])
b = tf.Variable(tf.zeros[10])

