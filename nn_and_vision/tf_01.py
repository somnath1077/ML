import tensorflow as tf

node1 = tf.constant(4.0, dtype=tf.float32)
node2 = tf.constant(5.0, dtype=tf.float32)

session = tf.Session()
#print(session.run([node1, node2]))

node3 = tf.add(node1, node2)
#print(session.run(node3))
#print(node1, node2)
#print(node3)

# placeholders

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b

#print(session.run(adder_node, {a: [1, 2, 3], b: [4, 3, 2]}))

add_and_triple = adder_node * 3
#print(session.run(add_and_triple, {a: [1, 2, 3], b: [4, 3, 2]}))

W = tf.Variable([0.30], dtype=tf.float32)
b = tf.Variable([-0.30], dtype=tf.float32)
x = tf.placeholder(dtype=tf.float32)

linear_model = W * x + b

init = tf.global_variables_initializer()
session.run(init)

y = tf.placeholder(dtype=tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

print(session.run(linear_model, {x: x_train, y: y_train}))
print(session.run(loss, {x: x_train, y: y_train}))

# Change the values of W and b to optimal ones, manually.

fixW = tf.assign(W, [-1.0])
fixb = tf.assign(b, [1.0])
session.run([fixW, fixb])
print(session.run(loss, {x: x_train, y: y_train}))

# Use Gradient Descent to find the optimal values of W and b

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

session.run(init) # reset values
for i in range(1000):
	session.run(train, {x: x_train, y: y_train})

print(session.run([W, b, loss], {x: x_train, y: y_train}))


