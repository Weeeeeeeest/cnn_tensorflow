import os

## Network config
##   Input width and Height
Height = 512
Width = 512

## Input variable size
##    If you use variable input size, below Variable_input is True.
##    Image is resized satisfying that longer side(width or height) is equal to Max_side.
##    If Variable_input is True, above "Height" and "Width" is ignored.
Variable_input = False
Max_side = 1024

Test_Max_side = 1024 #1536


# input data shape
# channels_last -> [mb, c, h, w] , channels_first -> [mb, h, w, c]
Input_type = 'channels_last'


## Class_label is corresponding to directory "name" including each labeled images.
Class_label = [
    'class1',
    'class2',
]

## Class_num is decided automatically from "Class_label".
Class_num = len(Class_label)

## Directory paths for training
Train_dirs = [
    'Data/Data1',
    'Data/Data2',
]

File_extensions = ['.jpg', '.png']

## Directory paths for test
Test_dirs = [
    'Data/Data3',
    'Data/Data4',
]



## Training config
Step = 10000
Minibatch = 8
Learning_rate = 0.0001

## Test config
##   if Minibatch is None, all data used for test
Test_Minibatch = None 


## Data augmentation
Horizontal_flip = True
Vertical_flip = True
Rotate_ccw90 = False


## Save config
Save_dir = 'models'
Model_name = 'cnn.h5'
Save_path = os.path.join(Save_dir, Model_name)



## Other config
##  Randon_seed is used for seed of dataset shuffle in data_loader.py
Random_seed = 0




## Check
variety = ['channels_first', 'channels_last']
if not Input_type in variety:
    raise Exception("unvalid Input_type")
