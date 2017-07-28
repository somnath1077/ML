import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# Placeholder variable to store an image (28 * 28 pixels = 784 pixels)
x = tf.placeholder(dtype=tf.float32, shape=[None, 784])

# Variables for the weight matrix and bias
W = tf.Variable(initial_value=tf.zeros([784, 10]))
b = tf.Variable(initial_value=tf.zeros([10]))

# The softmax model
# y = tf.nn.softmax(tf.matmul(x, W) + b)
y = tf.matmul(x, W) + b

# Loss function: cross-entropy
y_actual = tf.placeholder(dtype=tf.float32, shape=[None, 10])
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_actual * tf.log(y)), reduction_indices=[1])
# The numerically stable one
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_actual,
                                                                        logits=y))

# Training
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

session = tf.Session()
init = tf.global_variables_initializer()
session.run(init)

for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    session.run(train_step, feed_dict={x: batch_xs, y_actual: batch_ys})

# Evaluation

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_actual, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(session.run(accuracy, feed_dict={x: mnist.test.images, y_actual: mnist.test.labels}))





