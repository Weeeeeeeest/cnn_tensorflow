#!/usr/bin/env python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import config as cf
import argparse
import cv2
import numpy as np
import glob


from data_loader import DataLoader
#from vgg16 import NNN
from fcn import NNN



class Main_train():

    def __init__(self):
        pass


    def train(self):
        X = tf.placeholder(tf.float32, [None, None, None, 3])
        Y = tf.placeholder(tf.float32, [None, cf.Class_num])
        keep_prob = tf.placeholder(tf.float32)


        ## Load network model
        self.net = NNN(phase='Train')
        logits = self.net.model(X)
        #pred = tf.nn.softmax(train_model)

        ## Loss
        loss_op = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))

        ## Optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=cf.Learning_rate)
        train_op = optimizer.minimize(loss_op)

        ## Accuracy
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


        init = tf.global_variables_initializer()
        #init = tf.initialize_all_variables()

        ## Prepare Training data
        dl = DataLoader(phase='Train', shuffle=True)

        ## Prepare Test data
        dl_test = DataLoader(phase='Test', shuffle=True)
        if not cf.Variable_input:
            test_imgs, test_gts = dl_test.get_minibatch(shuffle=False)
            
        test_data_num = dl_test.get_data_num()
        
        ## PRepare Test data
        dl_test = DataLoader(phase='Test', shuffle=True)
        test_imgs, test_gts = dl_test.get_minibatch(shuffle=False)

        ## Start Train
        print('\n--------\nTraining Start!!')

        ## Secure GPU Memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(init)

            #saver = tf.train.Saver(max_to_keep=100)
            #self.net.load('pretrained_model/VGG_imagenet.ckpt', sess, saver, True)

            for step in range(cf.Step):
                step += 1

                imgs, gts = dl.get_minibatch(shuffle=True)

                sess.run(train_op, feed_dict={X: imgs, Y: gts, keep_prob: 0.5})


                if step % 10 == 0 or step == 1:
                    
                    test_num = cf.Test_Minibatch if cf.Test_Minibatch is not None else test_data_num
                    accuracy_count = 0
                    
                    for test_ind in range(test_num):
                        test_imgs, test_gts = dl_test.get_minibatch(shuffle=False)
                        pred = logits.eval(feed_dict={X: test_imgs, keep_prob: 1.0})[0]
                        test_gts = test_gts[0]
                        pred_label = np.argmax(pred)
                        test_label = np.argmax(test_gts)

                        if int(pred_label) == int(test_label):
                            accuracy_count += 1

                    accuracy = 1.0 * accuracy_count / test_num
                    
                    print('Step: {}, Accuracy: {:.4f} ({}/{})'.format(step, accuracy, accuracy_count, test_num))
                    #loss, acc = sess.run([loss_op, accuracy],
                    #                     feed_dict={X: test_imgs, Y: test_gts, keep_prob: 1.0})

                    #print('Step: {}, Loss: {}, Accuracy: {}'.format(step, loss, acc))

            ## Save trained model
            saver = tf.train.Saver()
            os.makedirs(cf.Save_dir, exist_ok=True)
            saver.save(sess, cf.Save_path)

            print('save model -> '.format(cf.Save_dir))


class Main_test():
    def __init__(self):
        pass


    def test(self):

        tf.reset_default_graph()

        X = tf.placeholder("float", [None, None, None, 3])
        Y = tf.placeholder("float", [None, cf.Class_num])
        keep_prob = tf.placeholder("float")

        ## Load network model
        self.net = NNN(phase='Test')
        logits = self.net.model(X)
        #probs = tf.nn.softmax(logits)


        ## Secure GPU Memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        print()
        print('Test start !!\n')

        table_gt_pred = np.zeros((cf.Class_num, cf.Class_num), dtype=np.int)

        with tf.Session(config=config) as sess:

            saver = tf.train.Saver()
            saver.restore(sess, cf.Save_path)

            img_paths = self.get_imagelist()

            for img_path in img_paths:

                img = self.get_image(img_path)
                gt = self.get_gt(img_path)

                pred = logits.eval(feed_dict={X: img, keep_prob: 1.0})[0]
                pred_label = np.argmax(pred)

                table_gt_pred[gt, pred_label] += 1

                #print('{} : {}'.format(img_path, pred))

        for cls_ind in range(cf.Class_num):
            print(cf.Class_label[cls_ind], np.round(table_gt_pred[cls_ind], 3))
            
        #print('gt-Real', np.round(table_gt_pred[0], 3))
        #print('gt-Synt', np.round(table_gt_pred[1], 3))

    def get_imagelist(self):
        dirs = cf.Test_dirs

        imgs = []

        for dir_path in dirs:
            img_list = glob.glob(dir_path + '/*')
            img_list.sort()
            imgs.extend(img_list)

        return imgs


    def get_image(self, img_path):

        img = cv2.imread(img_path).astype(np.float32)
        #img = cv2.resize(img, (cf.Width, cf.Height))
        
        if cf.Variable_input:
            longer_side = np.max(img.shape[:2])
            scaled_ratio = 1. * cf.Max_side / longer_side
            scaled_height = np.min([img.shape[0] * scaled_ratio, cf.Max_side]).astype(np.int)
            scaled_width = np.min([img.shape[1] * scaled_ratio, cf.Max_side]).astype(np.int)
            img = cv2.resize(img, (scaled_width, scaled_height))
        else:
            scaled_height = cf.Height
            scaled_width = cf.Width
            img = cv2.resize(img, (scaled_width, scaled_height))
            
        img = img[:, :, (2,1,0)]
        img = img[np.newaxis, :]
        img = img / 255.

        return img

    def get_gt(self, img_path):

        for ind, cls in enumerate(cf.Class_label):
            if cls in img_path:
                return ind

        raise Exception("Class label Error {}".format(img_path))
        

def arg_parse():
    parser = argparse.ArgumentParser(description='CNN implemented with Tensorflow')
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--test', dest='test', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = arg_parse()

    if args.train:
        main = Main_train()
        main.train()
    if args.test:
        main = Main_test()
        main.test()
