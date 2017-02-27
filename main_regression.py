# encoding: utf-8
import numpy as np
import PHM08
from sklearn.cross_validation import train_test_split
import tensorflow as tf
from sklearn.preprocessing import scale
import os


filename = './data/train.txt'

PHMs = []
units = set()
num_lines = 0

with open(filename, 'r') as f:
    temp = PHM08.PHM08()
    temp.unit = 1
    units.add(temp.unit)
    for line in f:
        num_lines += 1
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

    X_raw = np.empty((num_lines, 24))
    y_raw = np.empty((num_lines, 1))

    left = 0
    for phm in PHMs:
        X_temp, y_temp, n_cycles = phm.generate_data_for_regression()
        right = left + n_cycles
        X_raw[left:right, :] = X_temp
        y_raw[left:right, :] = y_temp
        left = right

X_raw = scale(X_raw)

n_inputs = 24
n_hidden = 4

x = tf.placeholder('float', [None, n_inputs])
y = tf.placeholder('float', [None, 1])

w_1 = tf.Variable(tf.random_normal([n_inputs, n_hidden]))
b_1 = tf.Variable(tf.random_normal([n_hidden]))

w_2 = tf.Variable(tf.random_normal([n_hidden, 1]))
b_2 = tf.Variable(tf.random_normal([1]))

h = tf.nn.softplus(tf.add(tf.matmul(x, w_1), b_1))
y_ = tf.nn.softplus(tf.add(tf.matmul(h, w_2), b_2))

loss = tf.reduce_mean(tf.square(y - y_))

optimizer = tf.train.MomentumOptimizer(learning_rate=0.0001, momentum=0.0001)
train = optimizer.minimize(loss)

if not os.path.exists('tmp/'):
    os.mkdir('tmp/')

sess = tf.Session()

saver = tf.train.Saver()

if os.path.exists('tmp/checkpoint'):
    saver.restore(sess, 'tmp/model.ckpt')
else:
    init = tf.initialize_all_variables()
    sess.run(init)

'''
y_loss = sess.run(y_, feed_dict={x: X_raw, y: y_raw})
sq = sess.run(tf.square(y - y_), feed_dict={x: X_raw, y: y_raw})

for i in range(450):
    print y_raw[i], y_loss[i], sq[i]

print sess.run(loss, feed_dict={x: X_raw, y: y_raw})

'''
for i in range(10000):
    _, loss_value = sess.run([train, loss], feed_dict={x: X_raw, y: y_raw})
    # print sess.run(y_, feed_dict={x: X_raw, y: y_raw})
    if i % 100 == 0:
        save_path = saver.save(sess, 'tmp/model.ckpt')
        print "模型保存:%s 当前训练损失:%s" % (save_path, np.sqrt(loss_value))
