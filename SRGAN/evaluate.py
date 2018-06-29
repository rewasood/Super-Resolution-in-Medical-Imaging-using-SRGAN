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


def evaluate(epoch, img_indx):
    ## create folders to save result images
    save_dir = "complete-data-test-evaluate-samples/{}".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir)

    #checkpoint_dir = "checkpoint"
    
    checkpoint_dir = "/Users/btopiwala/Downloads/CS231N/2018/Project/gcloud-run-all-data/checkpoint_epoch_20_epoch_48_with_intermediate_checkpoint/checkpoint"

    ###====================== PRE-LOAD DATA ===========================###
    # train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    # train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))

    ## If your machine have enough memory, please pre-load the whole train set.
    # train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    # for im in train_hr_imgs:
    #     print(im.shape)
    
    #valid_lr_imgs = tl.vis.read_images(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
    
    # for im in valid_lr_imgs:
    #     print(im.shape)
    
    #valid_hr_imgs = tl.vis.read_images(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)
    
    # for im in valid_hr_imgs:
    #     print(im.shape)
    # exit()

    
    valid_lr_imgs = []
    
    #sess = tf.Session()
    for img__ in valid_lr_img_list:
        
        
        image_loaded = scipy.misc.imread(os.path.join(config.VALID.lr_img_path, img__), mode='L')
        image_loaded = image_loaded.reshape((image_loaded.shape[0], image_loaded.shape[1], 1))

        valid_lr_imgs.append(image_loaded)
    
    print(type(valid_lr_imgs), len(valid_lr_img_list))
    
    
    
    valid_hr_imgs = []
    
    #sess = tf.Session()
    for img__ in valid_hr_img_list:
        
        
        image_loaded = scipy.misc.imread(os.path.join(config.VALID.hr_img_path, img__), mode='L')
        image_loaded = image_loaded.reshape((image_loaded.shape[0], image_loaded.shape[1], 1))

        valid_hr_imgs.append(image_loaded)
    
    print(type(valid_hr_imgs), len(valid_hr_img_list))
    
    
    
    
    
    ###========================== DEFINE MODEL ============================###
    
    #imid = 0  # 0: 企鹅  81: 蝴蝶 53: 鸟  64: 古堡
    
    #imid = img_indx    
    
    
    
    #t_image = tf.placeholder('float32', [1, None, None, 3], name='input_image')
    t_image = tf.placeholder('float32', [1, None, None, 1], name='input_image') # 1 for 1 channel

    net_g = SRGAN_g(t_image, is_train=False, reuse=False)

    ###========================== RESTORE G =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_srgan_{}.npz'.format(epoch), network=net_g)
    
    
    bicubic_psnrs = []
    psrns = []
    
    bicubic_ssims = []
    gen_ssims = []
    
    bicubic_ssims_with_data_range = []
    gen_ssims_with_data_range = []
    
    
    #for imid in range(len(valid_lr_imgs)):
    
    
    '''img_indx_sample = random.sample(range(len(valid_lr_imgs)), 30)
    filename = "img_ids.txt"
    with open(filename, "w") as myfile:
        myfile.write(','.join(str(v) for v in img_indx_sample))
    #sys.exit()
    for imid in img_indx_sample:
    '''
    for imid in [711,587,1141,1320,579,793,78,1197,480,1263,771,788,1097,994,715,1463,826,664,1099,414,403,1076,1389,27,756,1563,947]:
        
        valid_lr_img = valid_lr_imgs[imid]
        valid_hr_img = valid_hr_imgs[imid]
        # valid_lr_img = get_imgs_fn('test.png', 'data2017/')  # if you want to test your own image
        valid_lr_img = (valid_lr_img / 127.5) - 1  # rescale to ［－1, 1]
        # print(valid_lr_img.min(), valid_lr_img.max())

        size = valid_lr_img.shape
        # t_image = tf.placeholder('float32', [None, size[0], size[1], size[2]], name='input_image') # the old version of TL need to specify the image size



        ###======================= EVALUATION =============================###
        start_time = time.time()
        out = sess.run(net_g.outputs, {t_image: [valid_lr_img]})
        #print("took: %4.4fs" % (time.time() - start_time))

        #print("LR size: %s /  generated HR size: %s" % (size, out.shape))  # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
        #print("[*] save images")
        tl.vis.save_image(out[0], save_dir + '/valid_gen-id-' + str(imid) + "-epoch-" + str(epoch) + '.png')
        tl.vis.save_image(valid_lr_img, save_dir + '/valid_lr-id-'+ str(imid) +'.png')
        tl.vis.save_image(valid_hr_img, save_dir + '/valid_hr-id-'+ str(imid) +'.png')

        valid_lr_img = valid_lr_img.reshape(valid_lr_img.shape[0], valid_lr_img.shape[1])
        out_bicu = scipy.misc.imresize(valid_lr_img, [size[0] * 4, size[1] * 4], interp='bicubic') #, mode=None)
        tl.vis.save_image(out_bicu, save_dir + '/valid_bicubic-id-'+ str(imid) +'.png')

        hr_img_path = save_dir + '/valid_hr-id-'+ str(imid) +'.png'
        bi_cubic_img_path = save_dir + '/valid_bicubic-id-'+ str(imid) +'.png'
        gen_img_path = save_dir + '/valid_gen-id-' + str(imid) + "-epoch-" + str(epoch) + '.png'

        try:
            bicubic_psnr = computePSNR(hr_img_path, bi_cubic_img_path)
            gen_psnr = computePSNR(hr_img_path, gen_img_path)
            
            bicubic_psnrs.append(bicubic_psnr)
            psrns.append(gen_psnr)
            
            
            
            bicubic_ssim = computeSSIM(hr_img_path, bi_cubic_img_path)
            gen_ssim = computeSSIM(hr_img_path, gen_img_path)
            
            bicubic_ssims.append(bicubic_ssim)
            gen_ssims.append(gen_ssim)
            
            
            bicubic_ssim_with_data_range = computeSSIM_WithDataRange(hr_img_path, bi_cubic_img_path)
            gen_ssim_with_data_range = computeSSIM_WithDataRange(hr_img_path, gen_img_path)
            
            bicubic_ssims_with_data_range.append(bicubic_ssim_with_data_range)
            gen_ssims_with_data_range.append(gen_ssim_with_data_range)
            
            
            '''os.remove(hr_img_path)
            os.remove(save_dir + '/valid_lr-id-'+ str(imid) +'.png')
            os.remove(bi_cubic_img_path)
            os.remove(gen_img_path)
            '''

        except:
            print("imid:", imid)
            print("Unexpected error while computing psnr:", sys.exc_info()[0])

    
    filename = "psnr-generator-test-per-epoch.txt"
    with open(filename, "a") as myfile:
        myfile.write("%d,%f\n" % (epoch, np.mean(psrns)))
        
    filename = "psnr-bicubic-test-per-epoch.txt"
    with open(filename, "a") as myfile:
        myfile.write("%d,%f\n" % (epoch, np.mean(bicubic_psnrs)))    

    print("average bicubic psnr:", np.mean(bicubic_psnrs))
    print("average generator psnr:", np.mean(psrns))
    
    
    filename = "ssim-generator-test-per-epoch.txt"
    with open(filename, "a") as myfile:
        myfile.write("%d,%f\n" % (epoch, np.mean(gen_ssims)))
        
    filename = "ssim-bicubic-test-per-epoch.txt"
    with open(filename, "a") as myfile:
        myfile.write("%d,%f\n" % (epoch, np.mean(bicubic_ssims)))    

    print("average bicubic ssim:", np.mean(bicubic_ssims))
    print("average generator ssim:", np.mean(gen_ssims))
    
    
    filename = "ssim_with_data_range-generator-test-per-epoch.txt"
    with open(filename, "a") as myfile:
        myfile.write("%d,%f\n" % (epoch, np.mean(gen_ssims_with_data_range)))
        
    filename = "ssim_with_data_range-bicubic-test-per-epoch.txt"
    with open(filename, "a") as myfile:
        myfile.write("%d,%f\n" % (epoch, np.mean(bicubic_ssims_with_data_range)))    

    print("average bicubic ssim_with_data_range:", np.mean(bicubic_ssims_with_data_range))
    print("average generator ssim_with_data_range:", np.mean(gen_ssims_with_data_range))
    
    
    '''
    psnr_filename = "psnr-evaluate-generator-id-" + str(imid) + ".txt"
    bicubic_filename = "psnr-evaluate-bicubic-id-" + str(imid) + ".txt"

    with open(psnr_filename, "a") as myfile:
        myfile.write("epoch: %d. Generated image PSNR: %f\n" % (epoch, gen_psnr))

    with open(bicubic_filename, "a") as myfile:
        myfile.write("epoch: %d. Bicubic image PSNR: %f\n" % (epoch, bicubic_psnr))


    #with open("psnr_filename", "a") as myfile:
    #    myfile.write("epoch: %d. Bicubic image PSNR: %f\n" % (epoch, bicubic_psnr))
    #    myfile.write("epoch: %d. Generated image PSNR: %f\n" % (epoch, gen_psnr))

    print("epoch:", epoch, ". Bicubic image PSNR:", bicubic_psnr)
    print("epoch:", epoch, ". Generated image PSNR:", gen_psnr)    
    '''

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    #print(parser)
    
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--img_indx', type=int)
    
    args = parser.parse_args()
    
    print("args:", args)
    
    epoch = args.epoch
    img_indx = args.img_indx

    #parser.add_argument('--mode', type=str, default='srgan', help='srgan, evaluate')

    #args = parser.parse_args()

    tl.global_flag['mode'] = 'evaluate'

    #for i in range(3, 50, 3):
    #    epoch = i
    evaluate(epoch, img_indx)
