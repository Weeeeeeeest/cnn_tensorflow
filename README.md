# cnn_tensorflow

CNN examples implemented with Tensorflow-1.8(maybe you can use any version).

## Requirement

```
# For CPU only
$ pip install -r requirements_cpu.txt
# For GPU
$ pip install -r requirements_gpu.txt
```

## Setting
You cat set files and datasets as described below.

Datas are stored in "Data/Data*/" per categorized class.

For example, image files labeled as class1 are located in "Data/Data1",

images labeled as class2 are in "Data/Data2".

And, you set "Class_label" in "config.py" to class name which is contained in each image path.

e.g. If training file path is "Data/Data1/salamander.jpg", you can set  "salamander" or "Data1" as one of "Class_label".
```
cnn_tensorflow --- Data --- Data1 --- *.jpg(or png)
                |        |- Data2 --- *.jpg
```

## Training
When training, you change config.py responding to your environment.

You can change "Class_label", "Train_dirs" and "Test_dirs" in config.py.

"Train_dirs" and "Test_dirs" are sets of directory paths which include each images labeled.

"Class_label" is sets of directory names but not directory path.

If you train with your original dataset, please type below command.

```bash
$ python main.py --train
# For tf.contrib.slim
$ python main_slim.py --train
```

## Test
Whene testing, you change config.py.

The datails are as Training.

If you test with your original dataset, please type below command.

```bash
$ python main.py --test
# For tf.contrib.slim
$ python main_slim.py --test
```

## config.py
You can change class_labels, dataset paths, hyper-parameters(minibatch, learning rate and so on),
data augmentation.


## Network model
You can use VGG16 model or FCN model.

In main.py, you can change "from vgg16 import NNN" to "from fcn import NNN".


## Variable Input size
you can use variable input size image.

If you need, please change "Variable_input" in "config.py" to "True" and set "Max_side".
