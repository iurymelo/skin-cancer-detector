# Skin Cancer Detector
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) [![Version: Python](https://img.shields.io/badge/python-3.7.6-blue)](https://www.python.org/downloads/) [![Version: NumPy](https://img.shields.io/badge/numpy-1.18.1-blue)](https://docs.scipy.org/doc/numpy/user/install.html)

Python script to detect cancer on a model based on AlexNet. The dataset is too small to apply the full model, so it was simplified from over 63m paramters to over 6m. 
This is a project proposed by the [Machine Learning Engineer Nano Degree](https://www.udacity.com/course/machine-learning-engineer-nanodegree--nd009t).
The training process can be stopped and resumed. A PNG file is saved containing the loss and accuracy per epoch. This process allows the user to tune the learning rate between epochs. 


## Technologies :rocket: :

  * [Python](https://reactjs.org/)
  * [Keras](keras.io)
  * [NumPy](numpy.org)
 

## Network Summary
<h1 align="center">
<img src='https://i.imgur.com/7NNsaes.png'   alt="Summary" title="summary" />
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

If you want to make the imagem 100x100 grayscale, navigate back to the root folder an run:
```sh
python prepare-dataset.py
```

To start training run:
```sh
python train.py -c output/checkpoints
```

#### Resuming Training
By default, a snapshot is taken every 5 epochs. You can change this by updating the variable "backup_every" in 'train.py'.
To resume a stopped training run:
```
python train.py --checkpoints output/checkpoints --model output/checkpoints/epoch_'NUM_EPOCH'.hdf5 --start-epoch 'NUM_EPOCH'
```
Where 'NUM_EPOCH' is the number of the epoch.

### Validating
You can validate the model by running:
```sh
python validate --model output/checkpoints/name_model.hdf5 --dataset data/valid
```

**Made with :purple_heart: by Iury Melo !**
