import tensorflow as tf
import tensorlayer as tl
from tensorlayer.prepro import *
# from config import config, log_config
#
# img_path = config.TRAIN.img_path

import scipy
import numpy as np
import skimage

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
    
    #x[:, :, 1] = 0
    #x[:, :, 2] = 0

    
    #y = np.zeros((x.shape[0], x.shape[1], 3))  # Temporarily disabling
    #y[:,:,0] = x[:,:,0]  # Temporarily disabling

    
    #print("y.shape:", y.shape)
    
    
    #print("0", y[:,:,0])
    #print("1", y[:,:,1])
    #print("2", y[:,:,2])
    
    #print("after crop & padding channels:", x.shape)
    
    return x
    
    #return y # Temporarily disabling

def downsample_fn(x):
    # We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    #x = imresize(x, size=[96, 96], interp='bicubic', mode=None)
    
    #print("-->x.shape:", x.shape)
    #print("-->x.shape[-1]:", x.shape[-1])
    
    x = imresize(x, size=[56, 56], interp='bicubic', mode=None)
    x = x / (255. / 2.)
    x = x - 1.
    
    #x[:, :, 1] = 0
    #x[:, :, 2] = 0
    
    #y = np.zeros((x.shape[0], x.shape[1], 3)) # Temporarily disabling
    #y[:,:,0] = x[:,:,0] # Temporarily disabling
    
    #print("-->After x.shape:", x.shape)
    
    return x
    
    #print("-->After y.shape:", y.shape)
    
    #return y # Temporarily disabling

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