import tensorflow as tf
import time
import os
import numpy as np
import sys
from skimage import measure

from evaluate_utils import (
    input_setup,
    checkpoint_dir,
    read_data,
    merge,
    checkimage,
    imsave,
    crop_center,
    kaiming_normal,
    computeSSIM,
    computeSSIM_WithDataRange
)
class SRCNN(object):

    def __init__(self,
                 sess,
                 image_size,
                 label_size,
                 c_dim):
        self.sess = sess
        self.image_size = image_size
        self.label_size = label_size
        self.c_dim = c_dim
        self.build_model()

    def build_model(self):
        self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images')
        self.labels = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels')
        
        '''self.weights = {
            'w1': tf.Variable(tf.random_normal([9, 9, self.c_dim, 64], stddev=1e-3), name='w1'),
            'w2': tf.Variable(tf.random_normal([1, 1, 64, 32], stddev=1e-3), name='w2'),
            'w3': tf.Variable(tf.random_normal([5, 5, 32, self.c_dim], stddev=1e-3), name='w3')
        }
        '''
        
        #aa = np.zeros((9, 9, self.c_dim, 64))
        #print("aa shape:", aa.shape)

        self.weights = {
            'w1': tf.Variable(kaiming_normal((9, 9, self.c_dim, 64)), name="w1"),
            'w2': tf.Variable(kaiming_normal((1, 1, 64, 32)), name="w2"),
            
            #'w4': tf.Variable(kaiming_normal((3, 3, 32, self.c_dim)), name="w4"),
            #'w5': tf.Variable(kaiming_normal((3, 3, 32, self.c_dim)), name="w5"),
            ##'w6': tf.Variable(kaiming_normal((3, 3, 32, self.c_dim)), name="w6")
            
            'w3': tf.Variable(kaiming_normal((5, 5, 32, self.c_dim)), name="w3")
        }

        #W = tf.get_variable("W", shape=[784, 256], initializer=tf.contrib.layers.xavier_initializer())

        self.biases = {
            'b1': tf.Variable(tf.zeros([64], name='b1')),
            'b2': tf.Variable(tf.zeros([32], name='b2')),
            
            #'b4': tf.Variable(tf.zeros([32], name='b4')),
            #'b5': tf.Variable(tf.zeros([32], name='b5')),
            
            'b3': tf.Variable(tf.zeros([self.c_dim], name='b3'))
        }
        
        self.pred = self.model()
        
        #print("labels-pred:", self.labels, self.pred)
        self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))

        tf.summary.scalar('loss', self.loss)
        #self.merged_summary_op = tf.summary.merge_all()
        
        self.saver = tf.train.Saver() # To save checkpoint


    def model(self):
        #print("images:", self.images)
        
        ## VALID adds padding
        
        conv1 = tf.nn.relu(tf.nn.conv2d(self.images, self.weights['w1'], strides=[1,1,1,1], padding='VALID', name='conv1') + self.biases['b1'])
        
        #print("conv1:", conv1)
        
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, self.weights['w2'], strides=[1,1,1,1], padding='VALID', name='conv2') + self.biases['b2'])
        
        #print("conv2:", conv2)
        
        conv3 = tf.nn.conv2d(conv2, self.weights['w3'], strides=[1,1,1,1], padding='VALID', name='conv3') + self.biases['b3'] # This layer don't need ReLU
        
        #print("conv3:", conv3)
        
        return conv3


    def train(self, config, load_model_epoch, indx_, original_img, psrns, ssims, ssims_with_data_range):


        self.load(config.checkpoint_dir, load_model_epoch)


        config.is_train = False

        nx, ny, original_shape = input_setup(config)
        
        #print(" nx, ny, original_shape:",  nx, ny, original_shape)

        data_dir = checkpoint_dir(config)

        #print("data_dir:", data_dir)
        input_, label_ = read_data(data_dir)


        print("Now Start Testing...")
    
        result = self.pred.eval({self.images: input_})
            
        #print("result:", result.shape)
            
            
        #print(label_[1] - result[1])
        image = merge(result, [nx, ny], self.c_dim)
            
        #print("image after merge:", image.shape)
        '''print("[nx, ny]:", [nx, ny])

        print("original_shape:", original_shape)
            
        print(type(image), type(original_shape[0]), type(original_shape[1]))
        '''
        cropped_img = crop_center(image, original_shape[0], original_shape[1])
    
        #print("cropped_img:", cropped_img.shape)
            
        #image_LR = merge(input_, [nx, ny], self.c_dim)
        #checkimage(image_LR)
        
        #imsave(image, config.result_dir + '/result.png', config)
            
        #imsave(cropped_img, config.result_dir + '/result_crop.png', config)

        imsave(cropped_img, config.result_dir + '/srcnn-' + str(indx_) + '-epoch-' + str(load_model_epoch) + '.png', config)


        cropped_img = cropped_img * 255.
        cropped_img = cropped_img.astype(np.uint8)
        
        print(original_img.shape, original_img.dtype)
        print(cropped_img.shape, cropped_img.dtype)

        try:
            psnr = measure.compare_psnr(original_img, cropped_img)
            psrns.append(psnr)
            print(indx_, ", psnr:", psnr)
            

            ssim = computeSSIM(config.test_img, config.result_dir + '/srcnn-' + str(indx_) + '-epoch-' + str(load_model_epoch) + '.png')
            
            print(indx_, ", ssim:", ssim)
            ssims.append(ssim)
            

            ssim_with_data_range = computeSSIM_WithDataRange(config.test_img, config.result_dir + '/srcnn-' + str(indx_) + '-epoch-' + str(load_model_epoch) + '.png')
            
            print(indx_, ", ssim_with_data_range:", ssim_with_data_range)
            ssims_with_data_range.append(ssim_with_data_range)

            #os.remove(config.result_dir + '/srcnn-' + str(indx_) + '-epoch-' + str(load_model_epoch) + '.png')
            
        except:
            print("indx_:", indx_)
            print("Unexpected error while computing psnr / ssim:", sys.exc_info()[0])

            
    def load(self, checkpoint_dir, load_model_epoch):
        """
            To load the checkpoint use to test or pretrain
        """
        print("\nReading Checkpoints.....\n\n")
        model_dir = "%s_%s" % ("srcnn", self.label_size)# give the model name by label_size
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        
        model_file_path = os.path.join(checkpoint_dir, "SRCNN.model-" + str(load_model_epoch))

        #print("model_file_path:", model_file_path)
        if model_file_path:
            self.saver.restore(self.sess, os.path.join(os.getcwd(), model_file_path))
            print("\n Checkpoint Loading Success! %s\n\n"% model_file_path)
        else:
            print("\n! Checkpoint Loading Failed \n\n")
