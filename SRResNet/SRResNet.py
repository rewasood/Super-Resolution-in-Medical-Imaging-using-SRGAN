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

import scipy
import skimage
import scipy.misc
import skimage.measure

###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1

n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every

ni = int(np.sqrt(batch_size))


def train():
    ## create folders to save result images and trained model
    save_dir_gan = "samples/{}_gan".format(tl.global_flag['mode']) #srresnet
    tl.files.exists_or_mkdir(save_dir_gan)
    checkpoint_dir = "checkpoint" 
    tl.files.exists_or_mkdir(checkpoint_dir)

    ###====================== PRE-LOAD DATA ===========================###
    train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))

    ## If your machine has enough memory, please pre-load the whole train set.
    print("reading images")

    train_hr_imgs = []
    
    for img__ in train_hr_img_list:
        image_loaded = scipy.misc.imread(os.path.join(config.TRAIN.hr_img_path, img__), mode='L')
        image_loaded = image_loaded.reshape((image_loaded.shape[0], image_loaded.shape[1], 1))
        train_hr_imgs.append(image_loaded)
    
    print(type(train_hr_imgs), len(train_hr_img_list))
           
    ###========================== DEFINE MODEL ============================###
    ## train inference

    t_image = tf.placeholder('float32', [batch_size, 56, 56, 1], name='t_image_input_to_SRGAN_generator')
    t_target_image = tf.placeholder('float32', [batch_size, 224, 224, 1], name='t_target_image')

    print("t_image:", tf.shape(t_image))
    print("t_target_image:", tf.shape(t_target_image))
    
    net_g = SRGAN_g(t_image, is_train=True, reuse=False) #SRGAN_g is the SRResNet portion of the GAN
    
    print("net_g.outputs:", tf.shape(net_g.outputs))
    
    net_g.print_params(False)
    net_g.print_layers()

    ## vgg inference. 0, 1, 2, 3 BILINEAR NEAREST BICUBIC AREA
    t_target_image_224 = tf.image.resize_images(
        t_target_image, size=[224, 224], method=0,
        align_corners=False)  # resize_target_image_for_vgg # http://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/layers.html#UpSampling2dLayer
    t_predict_image_224 = tf.image.resize_images(net_g.outputs, size=[224, 224], method=0, align_corners=False)  # resize_generate_image_for_vgg

    
    ## Added as VGG works for RGB and expects 3 channels.
    t_target_image_224 = tf.image.grayscale_to_rgb(t_target_image_224)
    t_predict_image_224 = tf.image.grayscale_to_rgb(t_predict_image_224)
    
    print("net_g.outputs:", tf.shape(net_g.outputs))
    print("t_predict_image_224:", tf.shape(t_predict_image_224))
    
    net_vgg, vgg_target_emb = Vgg19_simple_api((t_target_image_224 + 1) / 2, reuse=False)
    _, vgg_predict_emb = Vgg19_simple_api((t_predict_image_224 + 1) / 2, reuse=True)

    ## test inference
    net_g_test = SRGAN_g(t_image, is_train=False, reuse=True)

    # ###========================== DEFINE TRAIN OPS ==========================###
    mse_loss = tl.cost.mean_squared_error(net_g.outputs, t_target_image, is_mean=True)
    vgg_loss = 2e-6 * tl.cost.mean_squared_error(vgg_predict_emb.outputs, vgg_target_emb.outputs, is_mean=True)

    g_loss = mse_loss + vgg_loss

    mse_loss_summary = tf.summary.scalar('Generator MSE loss', mse_loss)
    vgg_loss_summary = tf.summary.scalar('Generator VGG loss', vgg_loss)
    g_loss_summary = tf.summary.scalar('Generator total loss', g_loss)

    g_vars = tl.layers.get_variables_with_name('SRGAN_g', True, True)
    
    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)
    ## SRResNet
    g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars)

    ###========================== RESTORE MODEL =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_{}.npz'.format(tl.global_flag['mode']), network=net_g)
    
   ###============================= LOAD VGG ===============================###
    vgg19_npy_path = "vgg19.npy"
    if not os.path.isfile(vgg19_npy_path):
        print("Please download vgg19.npz from : https://github.com/machrisaa/tensorflow-vgg")
        exit()
    npz = np.load(vgg19_npy_path, encoding='latin1').item()

    params = []
    for val in sorted(npz.items()):
        W = np.asarray(val[1][0])
        b = np.asarray(val[1][1])
        print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
        params.extend([W, b])
    tl.files.assign_params(sess, params, net_vgg)

    ###============================= TRAINING ===============================###
    ## use first `batch_size` of train set to have a quick test during training
    sample_imgs = train_hr_imgs[0:batch_size]
    
    print("sample_imgs size:", len(sample_imgs), sample_imgs[0].shape)
    
    sample_imgs_224 = tl.prepro.threading_data(sample_imgs, fn=crop_sub_imgs_fn, is_random=False)
    print('sample HR sub-image:', sample_imgs_224.shape, sample_imgs_224.min(), sample_imgs_224.max())
    sample_imgs_56 = tl.prepro.threading_data(sample_imgs_224, fn=downsample_fn)
    print('sample LR sub-image:', sample_imgs_56.shape, sample_imgs_56.min(), sample_imgs_56.max())
    tl.vis.save_images(sample_imgs_56,[ni,ni], save_dir_gan + '/_train_sample_56.png')
    tl.vis.save_images(sample_imgs_224, [ni,ni], save_dir_gan + '/_train_sample_224.png')
    #tl.vis.save_image(sample_imgs_96[0],  save_dir_gan + '/_train_sample_96.png')
    #tl.vis.save_image(sample_imgs_384[0],save_dir_gan + '/_train_sample_384.png')
 
    ###========================= train SRResNet  =========================###
        
    merged_summary_generator = tf.summary.merge([mse_loss_summary, vgg_loss_summary, g_loss_summary]) #g_gan_loss_summary
    summary_generator_writer = tf.summary.FileWriter("./log/train/generator")
    
    learning_rate_writer = tf.summary.FileWriter("./log/train/learning_rate")
    
    count = 0
    for epoch in range(0, n_epoch + 1):
        ## update learning rate
        if epoch != 0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay**(epoch // decay_every)
            sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
            log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
            print(log)


            learning_rate_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="Learning_rate per epoch", simple_value=(lr_init * new_lr_decay)),]), (epoch))
            
            
        elif epoch == 0:
            sess.run(tf.assign(lr_v, lr_init))
            log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f (for GAN)" % (lr_init, decay_every, lr_decay)
            print(log)


            learning_rate_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="Learning_rate per epoch", simple_value=lr_init),]), (epoch))
            
            

        epoch_time = time.time()
        total_g_loss, n_iter = 0, 0

        
        
        ## If your machine cannot load all images into memory, you should use
        ## this one to load batch of images while training.
        # random.shuffle(train_hr_img_list)
        # for idx in range(0, len(train_hr_img_list), batch_size):
        #     step_time = time.time()
        #     b_imgs_list = train_hr_img_list[idx : idx + batch_size]
        #     b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=config.TRAIN.hr_img_path)
        #     b_imgs_384 = tl.prepro.threading_data(b_imgs, fn=crop_sub_imgs_fn, is_random=True)
        #     b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)

        ## If your machine have enough memory, please pre-load the whole train set.
        
        loss_per_batch = []
        
        mse_loss_summary_per_epoch = []
        vgg_loss_summary_per_epoch = []
        g_loss_summary_per_epoch = []
        
        
        for idx in range(0, len(train_hr_imgs), batch_size):
            step_time = time.time()
            b_imgs_224 = tl.prepro.threading_data(train_hr_imgs[idx:idx + batch_size], fn=crop_sub_imgs_fn, is_random=True)
            b_imgs_56 = tl.prepro.threading_data(b_imgs_224, fn=downsample_fn)    
            
            summary_pb = tf.summary.Summary()

            ## update G
            errG, errM, errV,  _, generator_summary = sess.run([g_loss, mse_loss, vgg_loss, g_optim, merged_summary_generator], {t_image: b_imgs_56, t_target_image: b_imgs_224}) #g_ga_loss
            
            

            summary_pb = tf.summary.Summary()
            summary_pb.ParseFromString(generator_summary)
            
            generator_summaries = {}
            for val in summary_pb.value:
            # Assuming all summaries are scalars.
                generator_summaries[val.tag] = val.simple_value
            
            mse_loss_summary_per_epoch.append(generator_summaries['Generator_MSE_loss'])
            vgg_loss_summary_per_epoch.append(generator_summaries['Generator_VGG_loss'])
            g_loss_summary_per_epoch.append(generator_summaries['Generator_total_loss'])
            
            print("Epoch [%2d/%2d] %4d time: %4.4fs, g_loss: %.8f (mse: %.6f vgg: %.6f)" %
                  (epoch, n_epoch, n_iter, time.time() - step_time, errG, errM, errV))
            total_g_loss += errG
            n_iter += 1

        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, g_loss: %.8f" % (epoch, n_epoch, time.time() - epoch_time / n_iter, total_g_loss / n_iter) 
        print(log)
        
        #####
        #
        # logging generator summary
        #
        ######

        summary_generator_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="Generator_MSE_loss per epoch", simple_value=np.mean(mse_loss_summary_per_epoch)),]), (epoch))
        
        summary_generator_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="Generator_VGG_loss per epoch", simple_value=np.mean(vgg_loss_summary_per_epoch)),]), (epoch))
        
        summary_generator_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="Generator_total_loss per epoch", simple_value=np.mean(g_loss_summary_per_epoch)),]), (epoch))
        
        out = sess.run(net_g_test.outputs, {t_image: sample_imgs_56}) 
        print("[*] save images")
        tl.vis.save_image(out[0],save_dir_gan + '/train_%d.png' % epoch)

        ## save model
        if (epoch != 0) and (epoch % 3 == 0):
        
            tl.files.save_npz(net_g.all_params, name=checkpoint_dir + '/g_{}_{}.npz'.format(tl.global_flag['mode'], epoch), sess=sess)

def evaluate():
    ## create folders to save result images
    save_dir = "samples/{}".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = "checkpoint"

    ###====================== PRE-LOAD DATA ===========================###
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))

    valid_lr_imgs = []
    
    for img__ in valid_lr_img_list:
        image_loaded = scipy.misc.imread(os.path.join(config.VALID.lr_img_path, img__), mode='L')
        image_loaded = image_loaded.reshape((image_loaded.shape[0], image_loaded.shape[1], 1))

        valid_lr_imgs.append(image_loaded)
    
    print(type(valid_lr_imgs), len(valid_lr_img_list))
    
    valid_hr_imgs = []
    
    for img__ in valid_hr_img_list:
        image_loaded = scipy.misc.imread(os.path.join(config.VALID.hr_img_path, img__), mode='L')
        image_loaded = image_loaded.reshape((image_loaded.shape[0], image_loaded.shape[1], 1))

        valid_hr_imgs.append(image_loaded)
    
    print(type(valid_hr_imgs), len(valid_hr_img_list))
    
    ###========================== DEFINE MODEL ============================###
    imid = 1 
    valid_lr_img = valid_lr_imgs[imid]
    valid_hr_img = valid_hr_imgs[imid]
    valid_lr_img = (valid_lr_img / 127.5) - 1  # rescale to ［－1, 1]
    
    size = valid_lr_img.shape
    t_image = tf.placeholder('float32', [1, None, None, 1], name='input_image') # 1 for 1 channel

    net_g = SRGAN_g(t_image, is_train=False, reuse=False)

    ###========================== RESTORE G =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_srgan.npz', network=net_g)

    ###======================= EVALUATION =============================###
    start_time = time.time()
    out = sess.run(net_g.outputs, {t_image: [valid_lr_img]})
    print("took: %4.4fs" % (time.time() - start_time))

    print("LR size: %s /  generated HR size: %s" % (size, out.shape))
    print("[*] save images")
    tl.vis.save_image(out[0], save_dir + '/valid_gen.png')
    tl.vis.save_image(valid_lr_img, save_dir + '/valid_lr.png')
    tl.vis.save_image(valid_hr_img, save_dir + '/valid_hr.png')

    valid_lr_img = valid_lr_img.reshape(valid_lr_img.shape[0], valid_lr_img.shape[1])
    out_bicu = scipy.misc.imresize(valid_lr_img, [size[0] * 4, size[1] * 4], interp='bicubic')
    tl.vis.save_image(out_bicu, save_dir + '/valid_bicubic.png')
    
    hr_img_path = save_dir + '/valid_hr.png'
    bi_cubic_img_path = save_dir + '/valid_bicubic.png'
    gen_img_path = save_dir + '/valid_gen.png'

    bicubic_psnr = computePSNR(hr_img_path, bi_cubic_img_path)
    gen_psnr = computePSNR(hr_img_path, gen_img_path)

    gnd_truth_hr_img=scipy.misc.imread(hr_img_path,mode='L')
    generated_hr_img=scipy.misc.imread(gen_img_path,mode='L')

    gen_ssim=skimage.measure.compare_ssim(gnd_truth_hr_img,generated_hr_img)

    print("Bicubic image PSNR:", bicubic_psnr)
    print("Generated image PSNR:", gen_psnr)
    print('Generated image SSIML', gen_ssim)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='srresnet', help='srresnet, evaluate')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'srresnet':
        train()
    elif tl.global_flag['mode'] == 'evaluate':
        evaluate()
    else:
        raise Exception("Unknown --mode")
