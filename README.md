# Skin Cancer Detector
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) [![Python](https://img.shields.io/badge/python-3.7.6-blue)] [![numpy](https://https://img.shields.io/badge/numpy-1.18.1-blue)]

Python script to detect cancer using the AlexNet model. This is a project proposed by the [Machine Learning Engineer Nano Degree](https://www.udacity.com/course/machine-learning-engineer-nanodegree--nd009t).
The training process can be stopped and resumed. A PNG file is saved containing the loss and accuracy per epoch. This process allows the user to tune the learning rate between epochs. 

## Technologies :rocket: :

  * [Python](https://reactjs.org/)
  * [Keras](keras.io)
  * [NumPy](numpy.org)
 

## Network Summary
<h1 align="center">
<img src='https://i.imgur.com/8LQZgjZ.png'   alt="Summary" title="summary" />
</h1>

## Setup
Clone this repo and navigate to the folder
```sh
git clone https://github.com/iurymelo/skin-cancer-detector.git
cd skin-cancer-detector 
```

### Training
Create a directory called data
```sh
mkdir data
cd data
```

Create folders to hold the training, validation, and test images.
```sh
mkdir train; mkdir valid; mkdir test
```

Download and unzip the [training data](https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/train.zip), [validation data](https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/valid.zip), and [test data](https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/test.zip).
Place the training, validation, and test images in the `data/` folder, at `data/train/`, `data/valid/`, and `data/test/`, respectively.

Navigate back to the root folder and prepare the data set by running:
```
python prepare-dataset.py
```

After the proccess is finished, run
```
python train.py -c output/checkpoints
```

#### Resuming Training
By default, a snapshot is taken every 5 epochs. You can change this by updating the variable "backup_every" in 'train.py'.
To resume a stopped training run:
```
python train.py --checkpoints output/checkpoints --model output/checkpoints/epoch_'NUM_EPOCH'.hdf5 --start-epoch 'NUM_EPOCH'
```
Where 'NUM_EPOCH' is the number of the epoch.

### Detecting Cancer
Work in progress :ghost::ghost::ghost:

## TODO
Next step is to 

**Made with :purple_heart: by Iury Melo !**
