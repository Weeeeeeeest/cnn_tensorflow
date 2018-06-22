import tensorflow as tf
import config as cf


class NNN():

    def __init__(self, phase='Train'):
        if phase == 'Train':
            self.train = True
        elif phase == 'Test':
            self.train = False
        else:
            raise Exception('Model phase not defined (phase="Train" or "Test")')
            


    def model(self, x):
        #x = tf.reshape(x, shape=[-1, None, None, 3])
        
        conv1_1 = tf.layers.conv2d(inputs=x, filters=32, kernel_size=[3,3], padding='same', activation=tf.nn.relu)
        conv1_2 = tf.layers.conv2d(inputs=conv1_1, filters=32, kernel_size=[3,3], padding='same', activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1_2, pool_size=[2,2], strides=2)

        conv2_1 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3,3], padding='same', activation=tf.nn.relu)
        conv2_2 = tf.layers.conv2d(inputs=conv2_1, filters=64, kernel_size=[3,3], padding='same', activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2_2, pool_size=[2,2], strides=2)

        conv3_1 = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=[3,3], padding='same', activation=tf.nn.relu)
        conv3_2 = tf.layers.conv2d(inputs=conv3_1, filters=128, kernel_size=[3,3], padding='same', activation=tf.nn.relu)
        conv3_3 = tf.layers.conv2d(inputs=conv3_2, filters=128, kernel_size=[3,3], padding='same', activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(inputs=conv3_3, pool_size=[2,2], strides=2)
        
        conv4_1 = tf.layers.conv2d(inputs=pool3, filters=256, kernel_size=[3,3], padding='same', activation=tf.nn.relu)
        conv4_2 = tf.layers.conv2d(inputs=conv4_1, filters=256, kernel_size=[3,3], padding='same', activation=tf.nn.relu)
        conv4_3 = tf.layers.conv2d(inputs=conv4_2, filters=256, kernel_size=[3,3], padding='same', activation=tf.nn.relu)
        pool4 = tf.layers.max_pooling2d(inputs=conv4_3, pool_size=[2,2], strides=2)

        conv5_1 = tf.layers.conv2d(inputs=pool4, filters=512, kernel_size=[3,3], padding='same', activation=tf.nn.relu)
        conv5_2 = tf.layers.conv2d(inputs=conv5_1, filters=512, kernel_size=[3,3], padding='same', activation=tf.nn.relu)
        conv5_3 = tf.layers.conv2d(inputs=conv5_2, filters=512, kernel_size=[3,3], padding='same', activation=tf.nn.relu)
        pool5 = tf.layers.max_pooling2d(inputs=conv5_3, pool_size=[2,2], strides=2)

        conv_class = tf.layers.conv2d(inputs=pool5, filters=cf.Class_num, kernel_size=[1,1], padding='same')

        out = conv_class
        
        ## Global avaerage pooling
        out = tf.reduce_mean(out, axis=1)
        ## 2nd reduce_mean is used to convert shape (1, 1, 2) to (1, 2)
        out = tf.reduce_mean(out, axis=1)
        
        pred = tf.nn.softmax(out)
        
        #last = pool5
        #mb, h, w, c = last.get_shape().as_list()
        #feature_shape = h * w * c
        #last_flat = tf.reshape(last, [-1, feature_shape])
        #fc1 = tf.layers.dense(inputs=last_flat, units=128, activation=tf.nn.relu)
        #fc1_drop = tf.layers.dropout(inputs=fc1, rate=0.5, training=self.train)
        #out = tf.layers.dense(inputs=fc1_drop, units=cf.Class_num)
        #pred = tf.nn.softmax(out)
        
        return pred

    
    def pred(self, x):
        out = self.model(x)
        pred = tf.nn.softmax(out)
        return pred
        
