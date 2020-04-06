# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 00:12:34 2020

@author: Iury Melo
@description: Machine Learning Algorithm to detect skin cancer in pictures. 
"""

# uncomment to force CPU use
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# force Agg backend
import matplotlib
matplotlib.use("Agg")

import cv2 as cv
import numpy as np
import keras.backend as K
import argparse
import sys

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer

from nn.resnet import ResNet
from utils.training_monitor import TrainingMonitor
from utils.epoch_checkpoint import EpochCheckpoint

import tensorflow as tf
import keras.backend.tensorflow_backend as tfback

def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

tfback._get_available_gpus = _get_available_gpus
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser() 
ap.add_argument("-c", "--checkpoints", required=True,
 	help="path to output checkpoint directory")
ap.add_argument("-m", "--model", type=str,
 	help="path to *specific* model checkpoint to load")
ap.add_argument("-s", "--start-epoch", type=int, default=0,
 	help="epoch to restart training at")
args = vars(ap.parse_args())


# initialize some parameters that will be used
num_epochs = 80
learning_rate = 1e-3
np.random.seed(8)
img_size = 200
dropout_value = 0.4
backup_every = 5
batch = 32

# optimizer
opt = Adam(learning_rate = learning_rate)

# create data generators and apply imgage augmentation for teste and train 
# datasets
train_datagen = ImageDataGenerator(horizontal_flip=True,
                                     fill_mode="nearest", 
                                     zca_whitening=False,
                                     )

test_datagem = ImageDataGenerator(horizontal_flip=True,
                                     fill_mode="nearest",
                                     zca_whitening=False)


# create batches of image to avoid memory overflow
train_batches = train_datagen.flow_from_directory('data/train/', 
                                                  class_mode='categorical',
                                                  target_size=(img_size, img_size), 
                                                  classes=['melanoma', 'nevus', 'seborrheic_keratosis'],
                                                  batch_size=batch)

test_batches = test_datagem.flow_from_directory('data/test/',
                                                class_mode='categorical',
                                                target_size=(img_size, img_size),
                                                classes=['melanoma', 'nevus', 'seborrheic_keratosis'],
                                                batch_size=batch)

# define alex net if no model is passed
if args["model"] is None:
	
    print("[INFO] compiling model...")
    model = Sequential()
    
    # layer 1
    model.add(Conv2D(filters=64, input_shape=(img_size, img_size, 3), 
                     kernel_size=(7, 7), strides=(4, 4), 
                     padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='valid',
                           dim_ordering="th"))
    
    #layer 2
    model.add(Conv2D(filters=64, kernel_size=(11, 11), 
                     strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='valid', 
                           dim_ordering="th"))
    
    # layer 3
    model.add(Conv2D(filters=128, kernel_size=(3, 3), 
                     strides=(2, 2), padding='valid'))
    model.add(Activation('relu'))
    
    # layer 4
    model.add(Conv2D(filters=128, kernel_size=(3, 3), 
                     strides=(2, 2), padding='valid'))
    model.add(Activation('relu'))
    
    # layer 5
    model.add(Conv2D(filters=256, kernel_size=(3, 3),
                     strides=(1, 1), padding='valid'))
    
    # Flatten
    model.add(Flatten())
    
    # full connected layer 1
    model.add(Dense(512, input_shape=(img_size*img_size*1,)))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_value))
    
    
    # output layer
    model.add(Dense(3))
    model.add(Activation('softmax'))
    
    # SUMMARY
    model.summary()
    
    
    # compile model
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    

# otherwise, we're using a checkpoint model
else:
	# load the checkpoint from disk
	print("[INFO] loading {}...".format(args["model"]))
	model = load_model(args["model"])

# build the path to the training plot and training history
plotPath = os.path.sep.join(["output", "skin_cancer.png"])
jsonPath = os.path.sep.join(["output", "skin_cancer.json"])

# construct the set of callbacks
callbacks = [
	EpochCheckpoint(args["checkpoints"], every=backup_every,
		startAt=args["start_epoch"]),
	TrainingMonitor(plotPath,
		jsonPath=jsonPath,
		startAt=args["start_epoch"])]

# train the network
print("[INFO] training network...")
model.fit_generator(
	train_batches,
	steps_per_epoch=64,
	epochs=num_epochs,
    validation_data=test_batches,
    validation_steps=12,
	callbacks=callbacks,
	verbose=1)

