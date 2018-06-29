import tensorflow as tf
from  evaluate import SRCNN
import os
import glob
import scipy
from tensorlayer.prepro import *
import tensorlayer as tl
import cv2
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("epoch", 10, "Number of epoch")
flags.DEFINE_integer("image_size", 33, "The size of image input")
flags.DEFINE_integer("label_size", 21, "The size of image output")
flags.DEFINE_integer("c_dim", 1, "The size of channel")
flags.DEFINE_boolean("is_train", True, "if the train")
flags.DEFINE_integer("scale", 4, "the size of scale factor for preprocessing input image")
flags.DEFINE_integer("stride", 21, "the size of stride")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory")
flags.DEFINE_float("learning_rate", 1e-4 , "The learning rate")
flags.DEFINE_integer("batch_size", 128, "the size of batch")
flags.DEFINE_string("result_dir", "result", "Name of result directory")
flags.DEFINE_string("test_img", "", "test_img")



def main(_): #?
    with tf.Session() as sess:
        
        #print("Calling init")

        srcnn = SRCNN(sess,
                      image_size = FLAGS.image_size,
                      label_size = FLAGS.label_size,
                      c_dim = FLAGS.c_dim)

        #print("Calling train")

        FLAGS.checkpoint_dir = "/Users/btopiwala/Downloads/CS231N/2018/Project/gcloud-run-all-data/srcnn/non-keras/checkpoint_complete_data_epoch_300"

        FLAGS.result_dir = "./psnr-per-epoch"
        #FLAGS.test_img = "./Test_LR/valid_hr_1.png"

        test_img_dir = "/Users/btopiwala/Downloads/CS231N/2018/Project/gcloud-run-all-data/data-used/complete_data%2Fcomplete_data/valid_HR"

        
        img_paths = sorted(tl.files.load_file_list(path=test_img_dir, regx='.*.png', printable=False))

        '''
        img_paths = glob.glob(os.path.join(test_img_dir, "*.png"))
        '''

        #img_paths = img_paths[0:3]

        
        for load_model_epoch in range(0, 500 + 1, 10): #500): #10):
            
            print("load_model_epoch:", load_model_epoch)
            

            psrns = []
            ssims = []
            ssims_with_data_range = []

            for indx_ in [711,587,1141,1320,579,793,78,1197,480,1263,771,788,1097,994,715,1463,826,664,1099,414,403,1076,1389,27,756,1563,947]:
            #[1289, 211, 482, 990, 1405]: # range(len(img_paths)):
                FLAGS.test_img = test_img_dir + "/" + img_paths[indx_]

                img = scipy.misc.imread(FLAGS.test_img, mode='L')
                img = img.reshape((img.shape[0], img.shape[1], 1))

                cv2.imwrite(FLAGS.result_dir + '/original-' + str(indx_) + '.png', img)

                srcnn.train(FLAGS, load_model_epoch, indx_, img, psrns, ssims, ssims_with_data_range)
                
            avg_psnr = np.mean(psrns)    
            filename = "psnr-per-epoch/" + "psnr-srcnn-test-per-epoch.txt"
            with open(filename, "a") as myfile:
                myfile.write(str(len(psrns)) + "," + str(load_model_epoch) + "," + str(avg_psnr) + "\n")
                
            
            avg_ssim = np.mean(ssims)    
            filename = "psnr-per-epoch/" + "ssim-srcnn-test-per-epoch.txt"
            with open(filename, "a") as myfile:
                myfile.write(str(len(psrns)) + "," + str(load_model_epoch) + "," + str(avg_ssim) + "\n")
                
                
                
            avg_ssim_with_data_range = np.mean(ssims_with_data_range)    
            filename = "psnr-per-epoch/" + "ssim-with-data-range-srcnn-test-per-epoch.txt"
            with open(filename, "a") as myfile:
                myfile.write(str(len(psrns)) + "," + str(load_model_epoch) + "," + str(avg_ssim_with_data_range) + "\n")      

if __name__=='__main__':
    tf.app.run() # parse the command argument , the call the main function
