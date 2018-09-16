import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

learning_rate = 0.001
training_epochs = 750
n_dim = 50
train_X = np.linspace(0, 1, n_dim)
train_Y = train_X * 3 + np.random.normal(0, 1.5, n_dim)

x = tf.placeholder(tf.float32, name="X")
y = tf.placeholder(tf.float32, name="Y")
W = tf.Variable(0.0, name="weight")
b = tf.Variable(0.0, name="bias")
linear_model = W * x + b
loss = tf.square(y - linear_model)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(training_epochs):
        sess.run(optimizer, feed_dict={x: train_X, y: train_Y})
    W_value, b_value = sess.run([W, b])

pred_X = train_X
pred_Y = pred_X * W_value + b_value

plt.plot(pred_X, pred_Y, color="r")
plt.scatter(train_X, train_Y)