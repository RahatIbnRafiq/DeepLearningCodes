import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

A = tf.placeholder(tf.float32, shape=(5,5), name='A')

v = tf.placeholder(tf.float32)

w = tf.matmul(A, v)

with tf.Session() as session:
    output = session.run(w, feed_dict={A:np.random.rand(5,5), v:np.random.rand(5,1)})
    print(output, type(output))


shape = (2, 2)
x = tf.Variable(tf.random_normal(shape))
t = tf.Variable(0) # a scalar

init = tf.global_variables_initializer()

with tf.Session() as session:
    out = session.run(init)
    print(out)

    print(x.eval())
    print(t.eval())


u = tf.Variable(20.0)
cost = u*u + 3*u + 4

train_op = tf.train.GradientDescentOptimizer(0.03).minimize(cost)
init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    for i in range(0, 12):
        session.run(train_op)
        print(" i = {}, cost = {} u = {}".format(i, cost.eval(), u.eval()))

