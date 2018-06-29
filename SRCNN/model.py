import tensorflow as tf
import time
import os
import numpy as np
import sys

from utils import (
    input_setup,
    checkpoint_dir,
    read_data,
    merge,
    checkimage,
    imsave,
    crop_center,
    kaiming_normal
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
        
        self.saver = tf.train.Saver(max_to_keep = 1000) # To save checkpoint


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

    def train(self, config):
        
        # NOTE : if train, the nx, ny are ingnored
        
        #print("config.is_train:", config.is_train)
        nx, ny, original_shape = input_setup(config)

        #print("nx, ny, original_shape:", nx, ny, original_shape)
        data_dir = checkpoint_dir(config)
        
        print("reading data..")
        input_, label_ = read_data(data_dir)
        
        print("input_", input_.shape)
        
        merged_summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter("./log/train_300") #, self.sess.graph)
        #self.summary_writer = tf.summary.FileWriter("./log/", tf.get_default_graph())

        
        # Stochastic gradient descent with the standard backpropagation
        #self.train_op = tf.train.GradientDescentOptimizer(config.learning_rate).minimize(self.loss)
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)
        #self.train_op = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(self.loss)
        tf.initialize_all_variables().run()

        counter = 0
        time_ = time.time()

        
        self.load(config.checkpoint_dir)
        # Train
        if config.is_train:
            print("Now Start Training...")
            #for ep in range(config.epoch):
                
            for ep in range(300, 1000+1, 1):   
                
                #print("ep:", ep)
                #sys.exit()
                
                loss_summary_per_batch = []
                
                
                
                # Run by batch images
                batch_idxs = len(input_) // config.batch_size
                for idx in range(0, batch_idxs):
                    batch_images = input_[idx * config.batch_size : (idx + 1) * config.batch_size]
                    batch_labels = label_[idx * config.batch_size : (idx + 1) * config.batch_size]
                    counter += 1
                    _, err, summary = self.sess.run([self.train_op, self.loss, merged_summary_op], feed_dict={self.images: batch_images, self.labels: batch_labels})

                    
                    summary_pb = tf.summary.Summary()
                    summary_pb.ParseFromString(summary)
                    
                    summaries = {}
                    for val in summary_pb.value:
                        summaries[val.tag] = val.simple_value

                    #print("summaries:", summaries)
                    
                    
                    loss_summary_per_batch.append(summaries['loss'])
                    
                    
                    summary_writer.add_summary(summary, (ep) * counter)

                    #self.summary_writer.add_summary(summary, (ep+1) * counter)
                    
                    if counter % 1000 == 0:
                        print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" % ((ep), counter, time.time()-time_, err))
                    #print(label_[1] - self.pred.eval({self.images: input_})[1],'loss:]',err)
                    
                    
                    #print("Epoch: [%2d], loss: [%.8f]", (ep+1), tf.reduce_mean(tf.square(label_ - self.pred.eval({self.images: input_}))))
                    
                    #if counter % 500 == 0:
                    #if counter % 20 == 0:
                    #    self.save(config.checkpoint_dir, counter)
                        
                if ep ==0 or ep % 10 == 0:
                    self.save(config.checkpoint_dir, ep)
                    
                    ###
                    '''
                    try:
                        config.is_train = False
                        nx_, ny_, original_shape_ = input_setup(config)
                        data_dir_ = checkpoint_dir(config)
                        input__, label__ = read_data(data_dir_)

                        

                        print("Now Start Testing...")

                        result_ = self.pred.eval({self.images: input__})                   
                        image_ = merge(result_, [nx_, ny_], self.c_dim)

                        print("image after merge:", image_.shape)
                        print("[nx_, ny_]:", [nx_, ny_])

                        print("original_shape:", original_shape_)

                        print(type(image__), type(original_shape_[0]), type(original_shape_[1]))
                        cropped_img_ = crop_center(image, original_shape_[0], original_shape_[1])

                        print("cropped_img_:", cropped_img_.shape)

                        imsave(image_, config.result_dir + '/result-' + ep + '.png', config)

                        imsave(cropped_img_, config.result_dir + '/result_crop-' + ep + '.png', config)
                    except:
                        print("Unexpected error while evaluating image:", sys.exc_info()[0])

                    config.is_train = True
                    '''

                    ###
                    
                print("loss per epoch[%d] loss: [%.8f]"  % ((ep), np.mean(loss_summary_per_batch)))
                summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="loss per epoch", simple_value=np.mean(loss_summary_per_batch)),]), ((ep)))
                
                summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="learning rate", simple_value=self.optimizer._lr),]), ((ep)))
                
                #print("learning rate:", self.optimizer._lr)
                
        # Test
        else:
            print("Now Start Testing...")
            #print("nx","ny",nx,ny)
            
            result = self.pred.eval({self.images: input_})
            
            print("result:", result.shape)
            
            
            #print(label_[1] - result[1])
            image = merge(result, [nx, ny], self.c_dim)
            
            print("image after merge:", image.shape)
            print("[nx, ny]:", [nx, ny])

            print("original_shape:", original_shape)
            
            print(type(image), type(original_shape[0]), type(original_shape[1]))
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
