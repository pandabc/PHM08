import numpy as np
import PHM08
from sklearn.cross_validation import train_test_split
import tensorflow as tf
from sklearn.preprocessing import scale

filename = './data/train.txt'

PHMs = []
units = set()

with open(filename, 'r') as f:
    temp = PHM08.PHM08()
    temp.unit = 1
    units.add(temp.unit)
    for line in f:
        cols = line.strip().split()
        unit = int(cols[0])
        if unit not in units:
            PHMs.append(temp)
            temp = PHM08.PHM08()
            temp.unit = unit
            units.add(temp.unit)
        temp.time.append(int(cols[1]))
        for i in range(3):
            temp.settings[i].append(float(cols[i+2]))
        for i in range(21):
            temp.sensors[i].append(float(cols[i+5]))
    PHMs.append(temp)

n_units = len(units)
n_samples = n_units*60
X_raw = np.empty((n_samples, 24))
Y_raw = np.empty((n_samples, 1))

for i in range(n_units):
    phm = PHMs[i]
    x, y = phm.generate_data()
    left = i*60
    right = i*60 + 60
    X_raw[left:right, :] = x
    Y_raw[left:right, :] = y

'''
for i in range(X_raw.shape[0]):
    for j in range(24):
        print X_raw[i, j],
    print ''

'''
X_raw = scale(X_raw)

X_train, X_test, y_train, y_test = train_test_split(X_raw, Y_raw, test_size=0.3, random_state=0)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

X_train = scale(X_train)

'''
for i in range(100):
    for j in range(24):
        print X_train[i, j],
    print ''

'''
n_samples = X_train.shape[0]
n_input = 24

x = tf.placeholder('float', [None, n_input])
y = tf.placeholder('float', [None, 1])

weights = tf.Variable(tf.random_normal([n_input, 1]))
bias = tf.Variable(tf.random_normal([1]))

y_ = tf.sigmoid(tf.add(tf.matmul(x, weights), bias))

loss = tf.reduce_mean(tf.square(y - y_))

optimizer = tf.train.GradientDescentOptimizer(0.7)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for _ in range(10000):
    print sess.run(loss, feed_dict={x: X_train, y: y_train})
    sess.run(train, feed_dict={x: X_train, y: y_train})
    # print sess.run(weights, feed_dict={x: X_train, y: y_train}), sess.run(bias, feed_dict={x: X_train, y: y_train})

print 1 - sum(abs(y_test - sess.run(y_, feed_dict={x: X_test})))/X_test.shape[0]
