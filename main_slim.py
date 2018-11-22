#!/usr/bin/env python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.contrib import slim

import argparse
import cv2
import numpy as np
import glob
import sys

import config as cf
from data_loader import DataLoader, load_image, get_gt
#from vgg16 import Model as Model
from model_slim_vgg16 import Model as Model
#from fcn import NNN as Model

class Main_train():
    def __init__(self):
        pass

    def train(self):
        X = tf.placeholder(tf.float32, [None, cf.Height, cf.Width, cf.Channel])
        Y = tf.placeholder(tf.float32, [None, cf.Class_num])
        keep_prob = tf.placeholder(tf.float32)

        ## Load network model
        predictions = Model(x=X)
        #logits = model.forward(X)
        #pred = tf.nn.softmax(train_model)

        ## Loss, Optimizer
        #loss_opt = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
        #loss = tf.reduce_mean(slim.losses.softmax_cross_entropy(predictions, Y))
        loss = tf.losses.softmax_cross_entropy(Y, predictions)
        opt = tf.train.AdamOptimizer(learning_rate=cf.Learning_rate)
        #total_loss = slim.losses.get_total_loss()
        #total_loss = tf.losses.get_total_loss()
        train_opt = slim.learning.create_train_op(loss, opt)
        #train_opt = opt.minimize(loss_opt)

        ## Accuracy
        correct_pred = tf.equal(tf.argmax(predictions, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        ## Prepare Training data
        dl = DataLoader(phase='Train', shuffle=True)
        
        ## Prepare Test data
        dl_test = DataLoader(phase='Test', shuffle=True)
        test_imgs, test_gts = dl_test.get_minibatch(shuffle=False)

        ## Start Train
        print('--------\nTraining Start!!')

        ## Secure GPU Memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            #saver = tf.train.Saver(max_to_keep=100)
            #self.net.load('pretrained_model/VGG_imagenet.ckpt', sess, saver, True)
            fname = os.path.join(cf.Save_dir, 'loss.txt')
            f = open(fname, 'w')
            f.write("Step,Train_loss,Test_loss" + os.linesep)

            saver = tf.train.Saver()

            for ite in range(1, cf.Iteration+1):
                x, y = dl.get_minibatch(shuffle=True)
                trainL, trainA = sess.run([train_opt, accuracy], feed_dict={X:x, Y:y, keep_prob:1.0})
                sess.run([train_opt], feed_dict={X:x, Y:y, keep_prob: 0.5})

                con = '|'
                if ite % cf.Save_train_step != 0:
                    for i in range(ite % cf.Save_train_step):
                        con += '>'
                    for i in range(cf.Save_train_step - ite % cf.Save_train_step):
                        con += ' '
                else:
                    for i in range(cf.Save_train_step):
                        con += '>'
                con += "| Iteration:{}, TrainL:{:.9f}, TrainA:{:.9f}".format(ite, trainL, trainA)

                if ite % cf.Save_train_step == 0 or ite == 1 or ite == cf.Iteration:
                    saver.save(sess, cf.Save_path)

                    test_num = dl_test.mb
                    #accuracy_count = 0.
                    test_x, test_y = dl_test.get_minibatch(shuffle=True)
                    """
                    for test_ind in range(test_num):
                        test_x, test_y = dl_test.get_minibatch(shuffle=False)
                        pred = logits.eval(feed_dict={X: test_x, keep_prob: 1.0})[0]
                        test_y = test_y[0]
                        pred_label = np.argmax(pred)
                        test_label = np.argmax(test_y)
                        if int(pred_label) == int(test_label):
                            accuracy_count += 1.
                    accuracy = accuracy_count / test_num
                    print('A:{:.9f} '.format(step, accuracy, accuracy_count, test_num))
                    """
                    L, A = sess.run([loss, accuracy],
                                         feed_dict={X: test_x, Y: test_y, keep_prob: 1.0})
                    con += ' TestL:{:.9f}, TestA:{:.9f}'.format(L, A)
                    con += '\n'
                    f.write("{},{},{},{},{}{}".format(ite, trainL, trainA, L, A, os.linesep))

                sys.stdout.write("\r"+con)
                    
            ## Save trained model
            saver.save(sess, cf.Save_path)
            print('\ntrained model was stored >> ', cf.Save_path)


class Main_test():
    def __init__(self):
        pass

    def test(self):
        tf.reset_default_graph()
        X = tf.placeholder("float", [None, None, None, cf.Channel])
        Y = tf.placeholder("float", [None, cf.Class_num])
        keep_prob = tf.placeholder("float")

        ## Load network model
        logits = Model(x=X, phase='Test')

        ## Secure GPU Memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        print('Test start !!')
        table_gt_pred = np.zeros((cf.Class_num, cf.Class_num), dtype=np.int)

        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, cf.Save_path)

            img_paths = get_imagelist()
            for img_path in img_paths:
                img = load_image(img_path)
                img = np.expand_dims(img, axis=0)
                gt = get_gt(img_path)
                pred = logits.eval(feed_dict={X: img, keep_prob: 1.0})[0]
                pred_label = np.argmax(pred)
                table_gt_pred[gt, pred_label] += 1
                print(img_path, pred)


        for cls_ind in range(cf.Class_num):
            print(cf.Class_label[cls_ind], np.round(table_gt_pred[cls_ind], 3))
            
        #print('gt-Real', np.round(table_gt_pred[0], 3))
        #print('gt-Synt', np.round(table_gt_pred[1], 3))

def get_imagelist():
    imgs = []
    for dir_path in cf.Test_dirs:
        for ext in cf.File_extensions:
            imgs += glob.glob(dir_path + '/*' + ext)
    imgs.sort()
    return imgs

def arg_parse():
    parser = argparse.ArgumentParser(description='CNN implemented with Tensorflow')
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--test', dest='test', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parse()

    if args.train:
        Main_train().train()
    if args.test:
        Main_test().test()

    if not (args.train or args.test):
        print("please select train or test flag")
        print("train: python main.py --train")
        print("test:  python main.py --test")
        print("both:  python main.py --train --test")

