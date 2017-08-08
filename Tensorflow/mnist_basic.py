import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

def get_data():
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    return mnist

def get_model_vars():
    # Variable to store an image (28 * 28 pixels = 784 pixels)
    x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
    # Variables for the weight matrix and bias
    W = tf.Variable(initial_value=tf.zeros([784, 10]))
    b = tf.Variable(initial_value=tf.zeros([10]))
    y_actual = tf.placeholder(dtype=tf.float32, shape=[None, 10])
    y = tf.matmul(x, W) + b

    return {'x': x,
            'y': y,
            'y_actual': y_actual,
            'W': W,
            'b': b}

def get_session():
    session = tf.Session()
    return session

def create_loss_function(y, y_actual):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_actual,
                                                                            logits=y))
    return cross_entropy

def train_model_and_evaluate(model_vars, loss_function, mnist, session):
    train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss_function)

    model_vars['loss_function'] = loss_function
    model_vars['train_step'] = train_step

    init = tf.global_variables_initializer()
    session.run(init)

    x = model_vars['x']
    y = model_vars['y']
    y_actual = model_vars['y_actual']

    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        session.run(train_step, feed_dict={x: batch_xs, y_actual: batch_ys})

    # Evaluation
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_actual, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy: ", session.run(accuracy, feed_dict={x: mnist.test.images, y_actual: mnist.test.labels}))


if __name__ == '__main__':
    mnist = get_data()
    model_vars = get_model_vars()
    session = get_session()
    loss_function = create_loss_function(model_vars['y'], model_vars['y_actual'])
    train_model_and_evaluate(model_vars, loss_function, mnist, session)


