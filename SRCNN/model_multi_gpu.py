import tensorflow as tf
import time
import os
import numpy as np
import keras
#from keras.models import Sequential
from keras.utils import multi_gpu_model
from keras.callbacks import TensorBoard
from time import time
from keras.models import load_model
#from keras.layers import *
#from keras.models import Model
import skimage.measure
from random import shuffle
import random
import sys

from utils_multi_gpu import (
    imread,
    make_sub_data,
    preprocess,
    input_setup,
    checkpoint_dir,
    read_data,
    merge,
    checkimage,
    imsave,
    crop_center,
    kaiming_normal
)

class MyCbk(keras.callbacks.Callback):

    def __init__(self, model, file_paths, config, c_dim):
         self.model_to_save = model
         self.file_paths = file_paths
         self.config = config
         self.c_dim = c_dim   

    def on_epoch_end(self, epoch, logs=None):
        
        if epoch == 0 or ((epoch+1) % 25 == 0):
            self.model_to_save.save('keras_model/model_at_epoch_%d.h5' % epoch)
            
            try:
                psnrs = []

                subset = 50

                ll = self.file_paths
                #shuffle(ll)

                #print("len:", len(ll[0:subset]))

                sub_ = random.sample(ll, subset)
                #print(sub_)
                for indx_ in range(len(sub_)): #len(ll)):

                    #print(indx_)
                    path = self.file_paths[indx_]
                    orig_img = imread(path)

                    padding = 0

                    paths = []
                    paths.append(path)

                    self.config.is_train = False

                    sub_input_sequence_, sub_label_sequence_, nx_, ny_, original_shape_, file_paths_ = make_sub_data(paths, padding, self.config)

                    self.config.is_train = True

                    # Make list to numpy array. With this transform
                    arrinput = np.asarray(sub_input_sequence_) # [?, 33, 33, 3]


                    out_put = self.model_to_save.predict(arrinput)

                    out_image = merge(out_put, [nx_, ny_], self.c_dim)

                    #print("path:", path)
                    #print("orig_img:", orig_img.shape)
                    #print("out_image:", out_image.shape)

                    cropped_img = crop_center(out_image, orig_img.shape[1], orig_img.shape[0])
                    tmp_img = 'tmp.png'
                    imsave(cropped_img, tmp_img, self.config)

                    #print("psnr:", skimage.measure.compare_psnr(orig_img.astype(np.float64), cropped_img))

                    psnrs.append(skimage.measure.compare_psnr(orig_img, imread(tmp_img)))
                    #print("psnr:", skimage.measure.compare_psnr(orig_img, imread(tmp_img)))
            except:
                print("Unexpected error while computing psnr:", sys.exc_info()[0])
            
            try:
                psnr_writer = tf.summary.FileWriter("./logs-keras/psnr")
                psnr_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="Average PSNR Train", simple_value=np.mean(psnrs)),]), epoch)

                print("average PSNR:", np.mean(psnrs), epoch)
            except:
                print("Unexpected error while adding average psnr summary:", sys.exc_info()[0])
            
            

class SRCNN(object):

    def __init__(self,
                 sess,
                 image_size,
                 label_size,
                 c_dim,
                 config):
        self.sess = sess
        self.image_size = image_size
        self.label_size = label_size
        self.c_dim = c_dim
        self.config = config
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

        '''
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
        ''' 
        
        
        #model = self.model()
        #model = multi_gpu_model(model, gpus=config.num_gpus)
        #model.compile(loss='mean_squared_error', optimizer='adam')
        
        
        
        self.pred = self.model()

        '''
        #print("labels-pred:", self.labels, self.pred)
        self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))

        tf.summary.scalar('loss', self.loss)
        #self.merged_summary_op = tf.summary.merge_all()
        
        
        self.saver = tf.train.Saver() # To save checkpoint
        '''

    def model(self):
        #print("images:", self.images)
        
        ## VALID adds padding
        ''' 
        conv1 = tf.nn.relu(tf.nn.conv2d(self.images, self.weights['w1'], strides=[1,1,1,1], padding='VALID', name='conv1') + self.biases['b1'])
        
        #print("conv1:", conv1)
        
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, self.weights['w2'], strides=[1,1,1,1], padding='VALID', name='conv2') + self.biases['b2'])
        
        #print("conv2:", conv2)
        
        conv3 = tf.nn.conv2d(conv2, self.weights['w3'], strides=[1,1,1,1], padding='VALID', name='conv3') + self.biases['b3'] # This layer don't need ReLU
        
        #print("conv3:", conv3)
        
        return conv3
        '''
    
        layers = [
            keras.layers.Conv2D(input_shape = (33, 33, 1), filters = 64, kernel_size=[9, 9], strides=1, padding='VALID', kernel_initializer=tf.keras.initializers.he_normal()),
            
            keras.layers.Conv2D(filters = 32, kernel_size=[1, 1], strides=1, padding='VALID', kernel_initializer=tf.keras.initializers.he_normal()),
            
            keras.layers.Conv2D(filters = 1, kernel_size=[5, 5], strides=1, padding='VALID', kernel_initializer=tf.keras.initializers.he_normal())
        ]

        model = keras.models.Sequential(layers) #tf.keras.Sequential(layers)


        self.original_model = model
        self.original_model.compile(loss='mean_squared_error', optimizer='adam')

        
        parallel_model = multi_gpu_model(self.original_model, gpus=self.config.num_gpus)
        parallel_model.compile(loss='mean_squared_error', optimizer='adam')
        
        
        print(model.summary())
        print(parallel_model.summary())
        
        
        return parallel_model

    def train(self, config):
        
        # NOTE : if train, the nx, ny are ingnored
        
        #print("config.is_train:", config.is_train)
        
        #print("skipping input_setup")
        nx, ny, original_shape, file_paths = input_setup(config)

        print("nx, ny, original_shape:", nx, ny, original_shape)
        data_dir = checkpoint_dir(config)
        
        print("reading data..")
        input_, label_ = read_data(data_dir)
        
        print("input_", input_.shape)
        
        '''
        merged_summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter("./log/train_multi_gpu") #, self.sess.graph)
        #self.summary_writer = tf.summary.FileWriter("./log/", tf.get_default_graph())
        '''
        
        # Stochastic gradient descent with the standard backpropagation
        #self.train_op = tf.train.GradientDescentOptimizer(config.learning_rate).minimize(self.loss)
        
        '''self.optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate)

        self.train_op = self.optimizer.minimize(self.loss, colocate_gradients_with_ops=True)

        #self.train_op = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(self.loss)
        tf.initialize_all_variables().run()

        counter = 0
        time_ = time.time()

        self.load(config.checkpoint_dir)
        '''

        # Train
        if config.is_train:
            print("Now Start Training...")
            
            #filepath = "model-{epoch:02d}.hdf5"
            #checkpoint = keras.callbacks.ModelCheckpoint(filepath, verbose=1, save_best_only=False, save_weights_only=False, period=1)
            
            cbk = MyCbk(self.original_model, file_paths, self.config, self.c_dim)
            
            tensorboard = TensorBoard(log_dir="logs-keras/{}".format(time()))
            #self.pred.fit(x = input_, y = label_, epochs=config.epoch, batch_size=config.batch_size, callbacks=[tensorboard, checkpoint])
            
            self.pred.fit(x = input_, y = label_, epochs=config.epoch, batch_size=config.batch_size, callbacks=[tensorboard, cbk])


            '''
            #print(file_end_index)

            psnrs = []
            
            for indx_ in range(len(file_paths)):
                
                path = file_paths[indx_]
 
                #print(len(input_[file_end_index[indx_] : file_end_index[indx_ + 1], :, :, :]))

                orig_img = imread(path)
                
                ####
            
                #print(path)
                padding = 0 #abs(config.image_size - config.label_size) / 2 <<-- For trainig

                
                #print(len(preprocess(path, self.config,  self.config.scale)))
                

                # Make sub_input and sub_label, if is_train false more return nx, ny
                
                paths = []
                paths.append(path)

                
                self.config.is_train = False
                
                sub_input_sequence_, sub_label_sequence_, nx_, ny_, original_shape_, file_paths_, file_end_index_ = make_sub_data(paths, padding, self.config)

                self.config.is_train = True
                
                # Make list to numpy array. With this transform
                arrinput = np.asarray(sub_input_sequence_) # [?, 33, 33, 3]
            
            
                ####
            
                out_put = self.pred.predict(arrinput) #input_[file_end_index[indx_] : file_end_index[indx_ + 1], :, :, :])
                
                #print("out_put:", out_put.shape)
                out_image = merge(out_put, [nx_, ny_], self.c_dim)
                
                #print(orig_img.shape)
                #print(out_image.shape)

                cropped_img = crop_center(out_image, orig_img.shape[0], orig_img.shape[1])
                tmp_img = 'tmp.png'
                imsave(cropped_img, tmp_img, self.config)
                
                #print(type(orig_img), type(cropped_img))                
                #print(orig_img.astype(np.float64).dtype, cropped_img.dtype)
                
                #print("psnr:", skimage.measure.compare_psnr(orig_img.astype(np.float64), cropped_img))
                
                psnrs.append(skimage.measure.compare_psnr(orig_img, imread(tmp_img)))
                #print("psnr:", skimage.measure.compare_psnr(orig_img, imread(tmp_img)))

            print("average PSNR:", np.mean(psnrs))'''
                
        # Test
        else:
            print("Now Start Testing...")
            #print("nx","ny",nx,ny)
            
            
            model = load_model('keras_model/model_at_epoch_9.h5')

            
            #result = self.pred.eval({self.images: input_})
            
            result = model.predict(input_)
            
            print("result:", result.shape)
            
            
            #print(label_[1] - result[1])
            image = merge(result, [nx, ny], self.c_dim)
            
            print("image after merge:", image.shape)
            print("[nx, ny]:", [nx, ny])

            print("original_shape:", original_shape)
            
            #print(type(image), type(original_shape[0]), type(original_shape[1]))
            cropped_img = crop_center(image, original_shape[0], original_shape[1])
            
            
            print("cropped_img:", cropped_img.shape)
            
            #image_LR = merge(input_, [nx, ny], self.c_dim)
            #checkimage(image_LR)
            imsave(image, config.result_dir+'/result.png', config)
            
            imsave(cropped_img, config.result_dir+'/result_crop.png', config)

           
            
    def load(self, checkpoint_dir):
        """
            To load the checkpoint use to test or pretrain
        """
        print("\nReading Checkpoints.....\n\n")
        model_dir = "%s_%s" % ("srcnn", self.label_size)# give the model name by label_size
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        
        # Check the checkpoint is exist 
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_path = str(ckpt.model_checkpoint_path) # convert the unicode to string
            self.saver.restore(self.sess, os.path.join(os.getcwd(), ckpt_path))
            print("\n Checkpoint Loading Success! %s\n\n"% ckpt_path)
        else:
            print("\n! Checkpoint Loading Failed \n\n")
    def save(self, checkpoint_dir, step):
        """
            To save the checkpoint use to test or pretrain
        """
        model_name = "SRCNN.model"
        model_dir = "%s_%s" % ("srcnn", self.label_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
             os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)
