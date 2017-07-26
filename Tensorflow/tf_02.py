import tensorflow as tf
import numpy as np

feature_columns = [tf.feature_column.numeric_column('x', shape=[1])]
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

x_train = np.array([1, 2, 3, 4])
y_train = np.array([0, -1, -2, -3])
x_eval = np.array([2, 5, 8, 1])
y_eval = np.array([-1.01, -4.1, -7, 0])

input_fn = tf.estimator.inputs.numpy_input_fn({'x': x_train},
                                              y_train,
                                              batch_size=4,
                                              num_epochs=1000)
eval_input_fn = tf.estimator.inputs.numpy_input_fn({'x': x_eval},
                                                   y_eval,
                                                   batch_size=4,
                                                   num_epochs=1000)

estimator.train(input_fn=input_fn, steps=1000)

train_loss = estimator.evaluate(input_fn=input_fn)
eval_loss = estimator.evaluate(input_fn=eval_input_fn)
print("Train loss: {}".format(train_loss))
print("Eval loss: {}".format(eval_loss))
