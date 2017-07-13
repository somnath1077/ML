import tensorflow as tf

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0, dtype=tf.float32)

# print(node1, node2)

session = tf.Session()
# print(session.run([node1, node2]))

node3 = tf.add(node1, node2)
# print("node3 = ", node3)
# print("session.run(node3) = ", session.run(node3))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder = a + b

conc_sess = session.run(adder, {a: 1, b: 2})
# print(conc_sess)

conc_sess2 = session.run(adder, {a: [1, 2, 3], b: [1, 2, 3]})
# print(conc_sess2)

add_and_triple = adder * 3

conc_sess3 = session.run(add_and_triple, {a: 2, b: 3})
# print(conc_sess3)

W = tf.Variable([0.3], dtype=tf.float32)
b = tf.Variable([-0.3], dtype=tf.float32)
x = tf.placeholder(dtype=tf.float32)
linear_model = W * x + b

init = tf.global_variables_initializer()
session.run(init)

y = tf.placeholder(dtype=tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

# fixW = tf.assign(W, [-1])
# fixb = tf.assign(b, [1])
# session.run([fixW, fixb])
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
session.run(init)
for i in range(1000):
    session.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

print(session.run([W, b]))
