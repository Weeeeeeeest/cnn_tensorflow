import tensorflow as tf
import config as cf
import numpy as np

"""
def conv2d(x, ksize=3, in_num=1, out_num=32, stride=1, bias=False, acti='relu'):
    W = tf.Variable(tf.random_normal([ksize, ksize, in_num, out_num]))
    x = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')
    if bias:
        b = tf.Variable(tf.random_normal([out_num]))
        x = tf.nn.bias_add(x, b)
    if acti == 'relu':
        return tf.nn.relu(x)
def maxpool2d(x, ksize=2, stride=2):
    return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding='SAME')
def fc(x, in_num=100, out_num=100, bias=False):
    W = tf.Variable(tf.random_normal([in_num, out_num]))
    x = tf.matmul(x, W)
    if bias:
        b = tf.Variable(tf.random_normal([out_num]))
        x = tf.add(x, b)
    return x
"""


def Model(x, phase):
    if phase == 'Train':
        train = True
    elif phase == 'Test':
        train = False
    else:
        raise Exception("model phase invalid >>", phase)

    inputs = tf.reshape(x, shape=[-1, cf.Height, cf.Width, cf.Channel])
    #x = tf.Variable(x)

    x = tf.layers.conv2d(inputs=inputs, filters=32, kernel_size=[3,3], padding='same', activation=tf.nn.relu, name='conv1_1')
    #x = tf.layers.conv2d(inputs=x, filters=32, kernel_size=[3,3], padding='same', activation=tf.nn.relu, name='conv1_2')
    x = tf.layers.max_pooling2d(inputs=x, pool_size=[2,2], strides=2)

    x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=[3,3], padding='same', activation=tf.nn.relu, name='conv2_1')
    x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=[3,3], padding='same', activation=tf.nn.relu, name='conv2_2')
    x = tf.layers.max_pooling2d(inputs=x, pool_size=[2,2], strides=2)
    last = x

    mb, h, w, c = last.get_shape().as_list()
    f_dim = h * w * c
    last_flat = tf.reshape(last, [-1, f_dim])
    
    x = tf.layers.dense(inputs=last_flat, units=100, activation=tf.nn.relu, name='fc6')
    x = tf.layers.dropout(inputs=x, rate=0.5, training=train)
    x = tf.layers.dense(inputs=x, units=cf.Class_num, name="cls_fc")
    
    pred = tf.nn.softmax(x)
    return pred

