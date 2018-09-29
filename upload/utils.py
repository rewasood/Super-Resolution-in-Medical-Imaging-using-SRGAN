import tensorflow as tf
import tensorlayer as tl
from tensorlayer.prepro import *
# from config import config, log_config
#
# img_path = config.TRAIN.img_path

import scipy
import numpy as np
import skimage

def Subpixel_mod(X, scale=2):
    I = X.outputs
    bsize, a, b, c = I.get_shape().as_list()
    X = tf.split(I, scale, 3)  # a, [bsize, b, r, r]
    X = tf.concat(X,2)  # bsize, b, a*r, r
    X = tf.reshape(X, (bsize, a*scale, b*scale, -1))
    return X

def Subpixel_anisotropic(X, scale=2):
    inp=X.outputs
    batch,h,w,c=inp.get_shape().as_list()
    n=tf.split(inp,scale,3)
   # print('after split: ' , n)
    n=tf.concat(n,2)
   # print('after concat: ', n)
    n=tf.reshape(n, (batch, h*scale, w, -1))
    return n

def get_imgs_fn(file_name, path):
    """ Input an image path and name, return an image array """
    # return scipy.misc.imread(path + file_name).astype(np.float)
    return scipy.misc.imread(path + file_name, mode='RGB')

def crop_sub_imgs_fn(x, is_random=True):
    #x = crop(x, wrg=384, hrg=384, is_random=is_random)
    
    #print("x.shape:", x.shape)
    x = crop(x, wrg=224, hrg=224, is_random=is_random)
    x = x / (255. / 2.)
    x = x - 1.
    
    return x
    
    #return y # Temporarily disabling

def downsample_fn(x):    
    x = imresize(x, size=[64, 64], interp='bicubic', mode=None)
    x = x / (255. / 2.)
    x = x - 1.
       
    return x
    
def downsample_fn_mod(x):
    x = imresize(x, size=[28,224], interp='bicubic', mode=None)
    x = x / (255. / 2. )
    x = x - 1.

    return x

def computePSNR(HR_img_path, to_compared_img_path):
    
    hr_img = scipy.misc.imread(HR_img_path, mode='L')
    to_be_compared_img = scipy.misc.imread(to_compared_img_path, mode='L')
    
    return skimage.measure.compare_psnr(hr_img, to_be_compared_img)


def computeSSIM(HR_img_path, to_compared_img_path):
    
    hr_img = scipy.misc.imread(HR_img_path, mode='L')
    to_be_compared_img = scipy.misc.imread(to_compared_img_path, mode='L')

    return skimage.measure.compare_ssim(hr_img, to_be_compared_img)


def computeSSIM_WithDataRange(HR_img_path, to_compared_img_path):
    
    hr_img = scipy.misc.imread(HR_img_path, mode='L')
    to_be_compared_img = scipy.misc.imread(to_compared_img_path, mode='L')

    return skimage.measure.compare_ssim(hr_img, to_be_compared_img, data_range=to_be_compared_img.max() - to_be_compared_img.min())
