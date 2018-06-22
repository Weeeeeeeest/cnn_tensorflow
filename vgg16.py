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


class NNN():

    def __init__(self, phase='Train'):
        if phase == 'Train':
            self.train = True
            self.mb = cf.Minibatch
        elif phase == 'Test':
            self.train = False
            self.mb = 1
        else:
            raise Exception('Model phase not defined (phase="Train" or "Test")')


    def load(self, data_path, session, saver, ignore_missing=False):
        if data_path.endswith('.ckpt'):
            saver.restore(session, data_path)
        else:
            data_dict = np.load(data_path).item()
            for key in data_dict:
                with tf.variable_scope(key, reuse=True):
                    for subkey in data_dict[key]:
                        try:
                            var = tf.get_variable(subkey)
                            session.run(var.assign(data_dict[key][subkey]))
                            print("assign pretrain model "+subkey+ " to "+key)
                        except ValueError:
                            print("ignore "+key)
                            if not ignore_missing:

                                raise
    """
    def model_old(self, x):
        x = tf.reshape(x, shape=[-1, cf.Height, cf.Width, 3])
        conv1_1 = conv2d(x, ksize=3, in_num=3, out_num=64, acti='relu', name='conv1_1')
        conv1_2 = conv2d(conv1_1, ksize=3, in_num=64, out_num=64, acti='relu', name='conv1_2')
        pool1 = maxpool2d(conv1_2, ksize=2, stride=2)
        conv2_1 = conv2d(pool1, ksize=3, in_num=64, out_num=128, acti='relu', name='conv2_1')
        conv2_2 = conv2d(conv2_1, ksize=3, in_num=128, out_num=128, acti='relu', name='conv2_2')
        pool2 = maxpool2d(conv2_2, ksize=2, stride=2)
        conv3_1 = conv2d(pool2, ksize=3, in_num=128, out_num=256, acti='relu', name='conv3_1')
        conv3_2 = conv2d(conv3_1, ksize=3, in_num=256, out_num=256, acti='relu', name='conv3_2')
        conv3_3 = conv2d(conv3_2, ksize=3, in_num=256, out_num=256, acti='relu', name='conv3_3')
        pool3 = maxpool2d(conv3_3, ksize=2, stride=2)
        conv4_1 = conv2d(pool3, ksize=3, in_num=256, out_num=512, acti='relu', name='conv4_1')
        conv4_2 = conv2d(conv4_1, ksize=3, in_num=512, out_num=512, acti='relu', name='conv4_2')
        conv4_3 = conv2d(conv4_2, ksize=3, in_num=512, out_num=512, acti='relu', name='conv4_3')
        pool4 = maxpool2d(conv4_3, ksize=2, stride=2)
        conv5_1 = conv2d(pool4, ksize=3, in_num=512, out_num=512, acti='relu', name='conv5_1')
        conv5_2 = conv2d(conv5_1, ksize=3, in_num=512, out_num=512, acti='relu', name='conv5_2')
        conv5_3 = conv2d(conv5_2, ksize=3, in_num=512, out_num=512, acti='relu', name='conv5_3')
        last = conv5_3
        mb, h, w, c = last.get_shape().as_list()
        feature_shape = h * w * c
        flat_last = tf.reshape(last, [-1, feature_shape])
	#fc6 = fc
        pred = fc(flat_last, in_num=feature_shape, out_num=cf.Class_num)
        #pred = conv2d(conv5_3, ksize=1, in_num=512, out_num=1)
        return pred
    """

    def model(self, x):
        x = tf.reshape(x, shape=[self.mb, cf.Height, cf.Width, 3])

        conv1_1 = tf.layers.conv2d(inputs=x, filters=32, kernel_size=[3,3], padding='same', activation=tf.nn.relu, name='conv1_1')
        conv1_2 = tf.layers.conv2d(inputs=conv1_1, filters=32, kernel_size=[3,3], padding='same', activation=tf.nn.relu, name='conv1_2')
        pool1 = tf.layers.max_pooling2d(inputs=conv1_2, pool_size=[2,2], strides=2)

        conv2_1 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3,3], padding='same', activation=tf.nn.relu, name='conv2_1')
        conv2_2 = tf.layers.conv2d(inputs=conv2_1, filters=64, kernel_size=[3,3], padding='same', activation=tf.nn.relu, name='conv2_2')
        pool2 = tf.layers.max_pooling2d(inputs=conv2_2, pool_size=[2,2], strides=2)

        conv3_1 = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=[3,3], padding='same', activation=tf.nn.relu, name='conv3_1')
        conv3_2 = tf.layers.conv2d(inputs=conv3_1, filters=128, kernel_size=[3,3], padding='same', activation=tf.nn.relu, name='conv3_2')
        conv3_3 = tf.layers.conv2d(inputs=conv3_2, filters=128, kernel_size=[3,3], padding='same', activation=tf.nn.relu, name='conv3_3')
        pool3 = tf.layers.max_pooling2d(inputs=conv3_3, pool_size=[2,2], strides=2)

        conv4_1 = tf.layers.conv2d(inputs=pool3, filters=256, kernel_size=[3,3], padding='same', activation=tf.nn.relu, name='conv4_1')
        conv4_2 = tf.layers.conv2d(inputs=conv4_1, filters=256, kernel_size=[3,3], padding='same', activation=tf.nn.relu, name='conv4_2')
        conv4_3 = tf.layers.conv2d(inputs=conv4_2, filters=256, kernel_size=[3,3], padding='same', activation=tf.nn.relu, name='conv4_3')
        pool4 = tf.layers.max_pooling2d(inputs=conv4_3, pool_size=[2,2], strides=2)

        conv5_1 = tf.layers.conv2d(inputs=pool4, filters=512, kernel_size=[3,3], padding='same', activation=tf.nn.relu, name='conv5_1')
        conv5_2 = tf.layers.conv2d(inputs=conv5_1, filters=512, kernel_size=[3,3], padding='same', activation=tf.nn.relu, name='conv5_2')
        conv5_3 = tf.layers.conv2d(inputs=conv5_2, filters=512, kernel_size=[3,3], padding='same', activation=tf.nn.relu, name='conv5_3')
        pool5 = tf.layers.max_pooling2d(inputs=conv5_3, pool_size=[2,2], strides=2)

        last = pool5

        mb, h, w, c = last.get_shape().as_list()
        feature_shape = h * w * c
        last_flat = tf.reshape(last, [-1, feature_shape])

        fc6 = tf.layers.dense(inputs=last_flat, units=4096, activation=tf.nn.relu, name='fc6')
        fc6_drop = tf.layers.dropout(inputs=fc6, rate=0.5, training=self.train)

        fc7 = tf.layers.dense(inputs=fc6_drop, units=4096, activation=tf.nn.relu, name='fc7')
        fc7_drop = tf.layers.dropout(inputs=fc7, rate=0.5, training=self.train)

        out = tf.layers.dense(inputs=fc7_drop, units=cf.Class_num, name="cls_fc")

        pred = tf.nn.softmax(out)
        print(pred.get_shape())
        return pred


    def pred(self, x):
        out = self.model(x)
        pred = tf.nn.softmax(out)
        return pred
