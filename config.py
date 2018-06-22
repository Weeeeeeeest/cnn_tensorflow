import os

## Network config
##   Input width and Height
Height = 128
Width = 128

## Input variable size
##    If you use variable input size, below Variable_input is True.
##    Image is resized satisfying that longer side(width or height) is equal to Max_side.
##    If Variable_input is True, above "Height" and "Width" is ignored.
Variable_input = True
Max_side = 240


## Class_label is corresponding to directory "name" including each labeled images.
Class_label = [
    'akahara',
    'yamori'
]

## Class_num is decided automatically from "Class_label".
Class_num = len(Class_label)


## Directory paths for training
Train_dirs = [
    'Data1/',
    'Data2/',
    'Data3/'
]

## Directory paths for test
Test_dirs = [
    'Data1/',
    'Data2/',
    'Data3/'
]


## Training config
Step = 300
Minibatch = 100
Learning_rate = 0.0001

## Test config
##   if Minibatch is None, all data used for test
Test_Minibatch = None 


## Data augmentation
Horizontal_flip = True
Vertical_flip = True

## Save config
Save_dir = 'out'
Model_name = 'CNN.ckpt'
Save_path = os.path.join(Save_dir, Model_name)



## Other config
##  Randon_seed is used for seed of dataset shuffle in data_loader.py
Random_seed = 0
