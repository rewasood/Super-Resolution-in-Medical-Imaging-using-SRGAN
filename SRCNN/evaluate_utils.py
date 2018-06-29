import cv2
import numpy as np
import tensorflow as tf
import scipy
import os 
import glob
import h5py
from skimage import measure


# Get the Image
def imread(path):
    #img = cv2.imread(path)
    img = scipy.misc.imread(path, mode='L')
    img = img.reshape((img.shape[0], img.shape[1], 1))
    #print("img shape:", img.shape)
    return img

def imsave(image, path, config):
    #checkimage(image)
    # Check the check dir, if not, create one
    if not os.path.isdir(os.path.join(os.getcwd(),config.result_dir)):
        os.makedirs(os.path.join(os.getcwd(),config.result_dir))

    # NOTE: because normial, we need mutlify 255 back    
    cv2.imwrite(os.path.join(os.getcwd(),path),image * 255.)

def checkimage(image):
    cv2.imshow("test",image)
    cv2.waitKey(0)

def modcrop(img, scale =3):
    """
        To scale down and up the original image, first thing to do is to have no remainder while scaling operation.
    """
    # Check the image is grayscale
    # print("before:", img.shape)
    if len(img.shape) ==3:
        h, w, _ = img.shape
        h = (h / scale) * scale
        w = (w / scale) * scale
        print("h:", h, "w:", w)
        img = img[0:h, 0:w, :]
    else:
        h, w = img.shape
        h = (h / scale) * scale
        w = (w / scale) * scale
        img = img[0:h, 0:w]
    #print("after:", img.shape)
    return img

def checkpoint_dir(config):
    if config.is_train:
        return os.path.join('./{}'.format(config.checkpoint_dir), "train.h5")
    else:
        #return os.path.join('./{}'.format(config.checkpoint_dir), "test.h5")
        return os.path.join('{}'.format(config.checkpoint_dir), "test.h5")

def getBorderSize():    
    s = 33 #21 # 33
    #print((float(H)/s), (H//s))
    
    border_size = s #((float(H)/s) - (H//s))*s #* 8

    border_size = int(border_size) / 2    
    
    return border_size
    
def preprocess(path, config, scale = 3):
    img = imread(path)

    a = img

    #print("path:", path)
    #cv2.imwrite(os.path.join(os.getcwd(),config.result_dir+'/input.png'), img)
    
    
    #print("img:", img.shape, "scale:", scale)

    label_ = modcrop(img, scale)
    
    #print("label_:", label_.shape)
    
    bicbuic_img = cv2.resize(label_,None,fx = 1.0/scale ,fy = 1.0/scale, interpolation = cv2.INTER_CUBIC)# Resize by scaling factor
    
    #print("bicbuic_img:", bicbuic_img.shape)
    
    input_ = cv2.resize(bicbuic_img,None,fx=scale ,fy=scale, interpolation = cv2.INTER_CUBIC)# Resize by scaling factor

    #print("input_:", input_.shape)
    
    #cv2.imwrite(os.path.join(os.getcwd(),config.result_dir+'/interpolated.png'), input_)
    
   
    '''
    H = input_.shape[0]
    
    s = 33 #21 # 33
    #print((float(H)/s), (H//s))
    
    border_size = s #((float(H)/s) - (H//s))*s #* 8

    border_size = int(border_size) / 2
    '''
    
    border_size = getBorderSize()


    #border_size = (12 / 2) # + 1 # Because we need overall padding 12 to be available <<-- Train
    
    #border_size = 16 # <<-- Test
    
    print("border_size:", border_size)
    
    #input_ = scipy.pad(input_, ((border_size,border_size),(border_size,border_size)), mode='reflect')
    if config.is_train == False:
        #input_ = scipy.pad(input_, ((border_size,border_size + 1),(border_size,border_size + 1)), mode='reflect')
        #input_ = scipy.pad(input_, ((border_size,border_size),(border_size,border_size)), mode='reflect')
        input_ = scipy.pad(input_, ((border_size,border_size),(border_size,border_size)), mode='reflect')
    
        #cv2.imwrite(os.path.join(os.getcwd(),config.result_dir+'/bordered.png'), input_)
    
        #print("shape matches:", a.shape, input_.shape)

    #print("shape matches:", input_.shape == a.shape)
    
    #input_ = cv2.resize(input_, (input_.shape[0] + 20, input_.shape[1] + 20))
    #print("input_:", input_.shape)

    return input_, label_, a.shape

def prepare_data(dataset="Train",Input_img=""):
    """
        Args:
            dataset: choose train dataset or test dataset
            For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp',..., 't99.bmp']
    """
    if dataset == "Train":
        data_dir = os.path.join(os.getcwd(), dataset) # Join the Train dir to current directory
        data = glob.glob(os.path.join(data_dir, "*.png")) # "*.bmp")) # make set of all dataset file path
    else:
        if Input_img !="":
            data = [os.path.join(os.getcwd(),Input_img)]
        else:
            data_dir = os.path.join(os.path.join(os.getcwd(), dataset), "Set5")
            data = glob.glob(os.path.join(data_dir, "*.png")) #"*.bmp")) # make set of all dataset file path
    return data

def load_data(is_train, test_img):
    if is_train:
        data = prepare_data(dataset="Train")
    else:
        if test_img != "":
            return prepare_data(dataset="Test",Input_img=test_img)
        data = prepare_data(dataset="Test")
    return data

def make_sub_data(data, padding, config):
    """
        Make the sub_data set
        Args:
            data : the set of all file path 
            padding : the image padding of input to label
            config : the all flags
    """
    sub_input_sequence = []
    sub_label_sequence = [] 
    for i in range(len(data)):
        if config.is_train:
            input_, label_, original_shape = preprocess(data[i], config, config.scale) # do bicbuic
        else: # Test just one picture
            input_, label_, original_shape = preprocess(data[i], config, config.scale) # do bicbuic
        
        if len(input_.shape) == 3: # is color
            h, w, c = input_.shape
        else:
            h, w = input_.shape # is grayscale
        #checkimage(input_)
    
    
        #print("label_:", label_.shape)

        #cnt = 0

        #print("h, w:", h, w)
        #print(0, h - config.image_size + 1, config.stride)

        #print(range(0, h - config.image_size + 1, config.stride), len(range(0, h - config.image_size + 1, config.stride)))
        #print(range(0, w - config.image_size + 1, config.stride), len(range(0, w - config.image_size + 1, config.stride)))
        
        #print(range(0, h, config.stride), len(range(0, h, config.stride)))

        nx, ny = 0, 0
        for x in range(0, h - config.image_size + 1, config.stride):
        #for x in range(0, h, config.stride):
            nx += 1; ny = 0
            for y in range(0, w - config.image_size + 1, config.stride):
            #for y in range(0, config.stride):
                ny += 1

                #print(x, x + config.image_size)

                sub_input = input_[x: x + config.image_size, y: y + config.image_size] # 33 * 33
                
                if config.is_train:
                
                    sub_label = label_[x + padding: x + padding + config.label_size, y + padding: y + padding + config.label_size] # 21 * 21

                
                #print(cnt, ":", x, x + config.image_size)
                #cnt = cnt + 1
                
                # Reshape the subinput and sublabel
                # print("input_ shape:", input_.shape)
                
                #print("sub_input shape:", sub_input.shape)

                ## Test code start
                '''if sub_input.shape[0] != 33:
                    print("sub_input shape:", sub_input.shape)
                    sub_input = np.vstack((sub_input, np.zeros((33 - sub_input.shape[0], 33))))
                    print("sub_input shape:", sub_input.shape)

                    
                if sub_label.shape[0] != 21:
                    print("sub_label shape:", sub_label.shape)
                    sub_label = np.vstack((sub_label, np.zeros((21 - sub_label.shape[0], 21, 1))))
                    print("sub_label shape:", sub_label.shape)    
                ''' 
                ## Test code end
                    
                #print("sub_input shape:", sub_input.shape)
                
                sub_input = sub_input.reshape([config.image_size, config.image_size, config.c_dim])
                
                #print(sub_label.shape)
                
                if config.is_train:
                    sub_label = sub_label.reshape([config.label_size, config.label_size, config.c_dim])
                    
                # Normialize
                sub_input =  sub_input / 255.0
                
                if config.is_train:
                    sub_label =  sub_label / 255.0

                #cv2.imshow("im1",sub_input)
                #cv2.imshow("im2",sub_label)
                #cv2.waitKey(0)

                # Add to sequence
                sub_input_sequence.append(sub_input)
                
                if config.is_train:
                    sub_label_sequence.append(sub_label)
                
                #print(sub_input.shape, sub_label.shape)

        
    # NOTE: The nx, ny can be ignore in train
    #print("nx, ny", nx, ny)
    #print(len(sub_input_sequence), len(sub_label_sequence))
    return sub_input_sequence, sub_label_sequence, nx, ny, original_shape


def read_data(path):
    """
        Read h5 format data file

        Args:
            path: file path of desired file
            data: '.h5' file format that contains  input values
            label: '.h5' file format that contains label values 
    """
    with h5py.File(path, 'r') as hf:
        input_ = np.array(hf.get('input'))
        label_ = np.array(hf.get('label'))
        return input_, label_

def make_data_hf(input_, label_, config):
    """
        Make input data as h5 file format
        Depending on "is_train" (flag value), savepath would be change.
    """
    # Check the check dir, if not, create one
    if not os.path.isdir(os.path.join(os.getcwd(),config.checkpoint_dir)):
        os.makedirs(os.path.join(os.getcwd(),config.checkpoint_dir))

    if config.is_train:
        savepath = os.path.join(os.getcwd(), config.checkpoint_dir + '/train.h5')
    else:
        savepath = os.path.join(os.getcwd(), config.checkpoint_dir + '/test.h5')

    with h5py.File(savepath, 'w') as hf:
        hf.create_dataset('input', data=input_)
        hf.create_dataset('label', data=label_)

def merge(images, size, c_dim):
    """
        images is the sub image set, merge it
    """
    h, w = images.shape[1], images.shape[2]
    
    print("h, w:", h, w)
    
    print("size[0], size[1]:", size[0], size[1])
    
    img = np.zeros((h*size[0], w*size[1], c_dim))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h : j * h + h,i * w : i * w + w, :] = image
        #cv2.imshow("srimg",img)
        #cv2.waitKey(0)
        
    return img

def input_setup(config):
    """
        Read image files and make their sub-images and saved them as a h5 file format
    """

    # Load data path, if is_train False, get test data
    data = load_data(config.is_train, config.test_img)

    padding = 0 #abs(config.image_size - config.label_size) / 2 <<-- For trainig
    
    print("padding:", padding)

    # Make sub_input and sub_label, if is_train false more return nx, ny
    sub_input_sequence, sub_label_sequence, nx, ny, original_shape = make_sub_data(data, padding, config)

    # Make list to numpy array. With this transform
    arrinput = np.asarray(sub_input_sequence) # [?, 33, 33, 3]
    arrlabel = np.asarray(sub_label_sequence) # [?, 21, 21, 3]

    make_data_hf(arrinput, arrlabel, config)

    return nx, ny, original_shape

'''def crop_center(img, cropx, cropy):
    y,x,_ = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)

    return img[starty:starty+cropy,startx:startx+cropx,:]
'''

def crop_center(img, cropx, cropy):
    y,x,_ = img.shape
    
    print(x-cropx, y-cropy)
    
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)

    
    print(y, cropy, y-cropy, int((y-cropy) / 2))
    print(x, cropx, x-cropx, int((x-cropx) / 2))
    
    #return img[int((y-cropy) / 2):y-int((y-cropy) / 2), int((x-cropx) / 2):x-int((x-cropx) / 2), :]

    #return img[y-cropy-4:y-4,x-cropx-4:x-4, :]
    
    border_size = getBorderSize()
    border_size = border_size // 2
    
    val = 4
    #return img[border_size + val: -border_size + val, border_size + val: -border_size + val, :]
    
    #return img[border_size + val: -border_size, border_size + val: -border_size, :]
    
    cropped_img = img[border_size + 9: , border_size+9: , :]
    if ((cropped_img.shape[0] + 1) == cropy) and ((cropped_img.shape[0] + 1) == cropx):
        print("re-cropped")
        cropped_img = img[border_size + 8: , border_size+8: , :]
    return cropped_img


def computeSSIM(HR_img_path, to_compared_img_path):
    
    hr_img = scipy.misc.imread(HR_img_path, mode='L')
    to_be_compared_img = scipy.misc.imread(to_compared_img_path, mode='L')

    #print(hr_img.shape, to_be_compared_img.shape)
    #print(to_be_compared_img.max(), to_be_compared_img.min())
    
    return measure.compare_ssim(hr_img, to_be_compared_img) #, data_range=to_be_compared_img.max() - to_be_compared_img.min())

def computeSSIM_WithDataRange(HR_img_path, to_compared_img_path):
    
    #print("computeSSIM_WithDataRange")
    
    hr_img = scipy.misc.imread(HR_img_path, mode='L')
    to_be_compared_img = scipy.misc.imread(to_compared_img_path, mode='L')

    #print("images loaded")
    
    #print(hr_img.shape, to_be_compared_img.shape)
    #print("-->", type(to_be_compared_img), type(to_be_compared_img))
    #print("-->", to_be_compared_img.max(), to_be_compared_img.min())
    
    return measure.compare_ssim(hr_img, to_be_compared_img, data_range=to_be_compared_img.max() - to_be_compared_img.min())

def kaiming_normal(shape):
        if len(shape) == 2:
            fan_in, fan_out = shape[0], shape[1]
        elif len(shape) == 4:
            fan_in, fan_out = np.prod(shape[:3]), shape[3]
        return tf.random_normal(shape) * np.sqrt(2.0 / fan_in)