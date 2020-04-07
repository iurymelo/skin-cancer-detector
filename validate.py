# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 16:03:10 2020

@author: iurym
"""

# Imports
import argparse
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

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

# Define image size and batch size
img_size = 200
batch = 32

'''
Parse the arguments
--model: Path to the model to be validated
--dataset: Path where the dataset to be used to validate the model is located
'''
ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', required=True, 
                help='Path to the model to be validated')
ap.add_argument('-d', '--dataset', required=True,
                 help='Path where the dataset is located')
args = vars(ap.parse_args())

# Load model and output path
model = load_model(args['model'])
dataset = args['dataset']

# Create a generator
valid_gen = ImageDataGenerator(zca_whitening=False)
# Batch the files to save memory
valid_batch = valid_gen.flow_from_directory(dataset, class_mode='categorical',
                                                  target_size=(img_size, img_size), 
                                                  classes=['melanoma', 'nevus', 'seborrheic_keratosis'],
                                                  batch_size=batch)

res = model.evaluate_generator(valid_batch)
print("Validation Completed!")
print('Loss: {}, Acc: {}'.format(res[0], res[1]))
