# cnn_tensorflow

CNN examples implemented with Tensorflow-1.5 (1.8).

## Requirement

```
Python 3   (because I used os.makedirs(exist_ok) in main.py)
tensorflow 1.5
opencv-python
numpy
```

## Setting
You cat set files and datasets as described below.

Datas are stored in Data/Data*/ per categorized class.

For example, image files labeled as class1 are located in Data/Data1,

images labeled as class2 are in Data/Data2.
```
cnn_tensorflow --- Data --- Data1 --- *.jpg
                |        |- Data2 --- *.jpg
                |
                |- config.py
                |- data_loader.py
                |- fcn.py
                |- main.py
                |- vgg16.py
```

## Training
When training, you change config.py responding to your environment.

You can change "Class_label", "Train_dirs" and "Test_dirs" in config.py.

"Train_dirs" and "Test_dirs" are sets of directory paths which include each images labeled.

"Class_label" is sets of directory names but not directory path.

If you train with your original dataset, please type below command.
```
python main.py --train
```

## Test
Whene testing, you change config.py.

The datails are as Training.

If you test with your original dataset, please type below command.
```
python main.py --test
```

## config.py
You can change class_labels, dataset paths, hyper-parameters(minibatch, learning rate and so on),
data augmentation.


## Network model
You can use VGG16 model or FCN model.

In main.py, you can change "from vgg16 import NNN" to "from fcn import NNN".
