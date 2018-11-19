import os

## Network config
##   Input width and Height
Height = 64
Width = 64
Channel = 3

## Input variable size
##    If you use variable input size, below Variable_input is True.
##    Image is resized satisfying that longer side(width or height) is equal to Max_side.
##    If Variable_input is True, above "Height" and "Width" is ignored.
Variable_input = False
Max_side = 512

# input data shape
# channels_last -> [mb, c, h, w] , channels_first -> [mb, h, w, c]
Input_type = 'channels_last'


## Class_label is corresponding to directory "name" including each labeled images.
Class_label = ['imori', 'yamori',]

## Class_num is decided automatically from "Class_label".
Class_num = len(Class_label)

File_extensions = ['.jpg', '.png']

## Directory paths for training
Train_dirs = [
    '/Users/yoshito/work_space/Dataset/train/imori',
    '/Users/yoshito/work_space/Dataset/train/yamori',
]

## Directory paths for test
Test_dirs = [
    '/Users/yoshito/work_space/Dataset/test/imori',
    '/Users/yoshito/work_space/Dataset/test/yamori',
]

## Training config
Iteration = 50
Minibatch = 128
Learning_rate = 0.001

## Test config
##   if Minibatch is None, all data used for test
Test_Minibatch = 10

## Data augmentation
Horizontal_flip = True
Vertical_flip = True
Rotate_ccw90 = False

Save_train_step = 20
Save_iteration_disp = True

## Save config
Save_dir = 'models'
Model_name = 'model.h5'
Save_path = os.path.join(Save_dir, Model_name)

## Other config
##  Randon_seed is used for seed of dataset shuffle in data_loader.py
Random_seed = 0

## Check
variety = ['channels_first', 'channels_last']
if not Input_type in variety:
    raise Exception("unvalid Input_type")
if not os.path.exists(Save_path):
    os.makedirs(Save_path)
