import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

df = pd.read_csv('data/Iris.csv')
df = shuffle(df)
x = df.values[:, 0:4]
y = pd.get_dummies(df, prefix=['Iris']).values[:, 4:7]

x_data = np.array(x[0:int(x.shape[0]*0.8), :])
y_data = np.array(y[0:int(y.shape[0]*0.8), :])

x_test = np.array(x[int(x.shape[0]*0.8):, :])
y_test = np.array(y[int(y.shape[0]*0.8):, :])

X = tf.placeholder(tf.float32, [None, 4])
Y = tf.placeholder(tf.float32, [None, 3])

W = tf.Variable(tf.random_normal([4, 3]))
b = tf.Variable(tf.random_normal([3]))

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
loss = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=10e-2)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1)), tf.float32))

train = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(501):
        cost_val, _ = sess.run([loss, train], feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:
            print(step, cost_val)
            print(hypothesis.eval(session=sess, feed_dict={X: x_test, Y: y_test}))
    print("Accuracy: ", accuracy.eval(session=sess, feed_dict={X: x_test, Y: y_test}))