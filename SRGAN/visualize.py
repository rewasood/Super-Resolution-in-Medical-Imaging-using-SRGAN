#! /usr/bin/python
# -*- coding: utf8 -*-

import os, time, pickle, random, time
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy

import tensorflow as tf
import tensorlayer as tl
from model import SRGAN_g, SRGAN_d, Vgg19_simple_api
from utils import *
from config import config, log_config
import sys
import random

###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
## initialize G
n_epoch_init = config.TRAIN.n_epoch_init
## adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every

ni = int(np.sqrt(batch_size))


def visualize(epoch):

    #checkpoint_dir = "checkpoint"
    
    checkpoint_dir = "/Users/btopiwala/Downloads/CS231N/2018/Project/gcloud-run-all-data/checkpoint_epoch_20_epoch_48_with_intermediate_checkpoint/checkpoint"

    
    ###========================== DEFINE MODEL ============================###


    #t_image = tf.placeholder('float32', [1, None, None, 3], name='input_image')
    t_image = tf.placeholder('float32', [1, None, None, 1], name='input_image') # 1 for 1 channel

    net_g = SRGAN_g(t_image, is_train=False, reuse=False)

    ###========================== RESTORE G =============================###

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_srgan_{}.npz'.format(epoch), network=net_g)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    epoch = 48 # args.epoch

    visualize(epoch)
