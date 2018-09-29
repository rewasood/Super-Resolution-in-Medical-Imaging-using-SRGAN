from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()

## Adam
config.TRAIN.batch_size = 16 #8 #16
config.TRAIN.lr_init = 1e-4
config.TRAIN.beta1 = 0.9

## initialize G
config.TRAIN.n_epoch_init = 20
    # config.TRAIN.lr_decay_init = 0.1
    # config.TRAIN.decay_every_init = int(config.TRAIN.n_epoch_init / 2)

## adversarial learning (SRGAN)
config.TRAIN.n_epoch = 40
config.TRAIN.lr_decay = 0.1
config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 2)

#print("config.TRAIN.decay_every:", config.TRAIN.decay_every)

## train set location
config.TRAIN.hr_img_path = '../SRGAN8x/DATA/train_HR/' #'data2017/DIV2K_train_HR/'
config.TRAIN.lr_img_path = '../SRGAN8x/DATA/train_LR/'#aniso/' #'data2017/DIV2K_train_LR_bicubic/X4/'


#config.TRAIN.hr_img_path = 'data2017/DIV2K_train_HR/'
#config.TRAIN.lr_img_path = 'data2017/DIV2K_train_LR_bicubic/X4/'

config.VALID = edict()
## test set location
config.VALID.hr_img_path = '../SRGAN8x/DATA/valid_HR/' #'data2017/DIV2K_valid_HR/'
config.VALID.lr_img_path = '../SRGAN8x/DATA/valid_LR/' #'data2017/DIV2K_valid_LR_bicubic/X4/'

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
