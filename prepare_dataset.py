# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 02:06:50 2020

@author: iurym
"""
import cv2 as cv
import os
import numpy as np

train_images = []
train_labels = []
test_images =  []
test_labels = []

base_path = 'C:/Users/iurym/Documents/Python/DermatologistAI/dermatologist-ai/dermatologis-ai/'
melanoma_train_dir ='data/train/melanoma/'
nevus_train_dir = 'data/train/nevus/'
seborrheic_keratosis_train_dir = 'data/train/seborrheic_keratosis/'
melanoma_test_dir = 'data/test/melanoma/'
nevus_test_dir = 'data/test/nevus/'
seborrheic_keratosis_test_dir = 'data/test/seborrheic_keratosis/'

print("[INFO] INITIALIZING DATASET PREPARATION...")
print("[INFO] this proccess may take a while to complete...")
print("[INFO] preparaing train set")

for filename in os.listdir(melanoma_train_dir):
    image = cv.imread(base_path+melanoma_train_dir+filename)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = cv.resize(image, (100, 100), interpolation=cv.INTER_AREA)
    train_images.append(image)
    train_labels.append('melanoma')


for filename in os.listdir(nevus_train_dir):
    image = cv.imread(base_path+nevus_train_dir+filename)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = cv.resize(image, (100, 100), interpolation=cv.INTER_AREA)
    train_images.append(image)
    train_labels.append('nevus')
    
for filename in os.listdir(seborrheic_keratosis_train_dir):
    image = cv.imread(base_path+seborrheic_keratosis_train_dir+filename)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = cv.resize(image, (100, 100), interpolation=cv.INTER_AREA)
    train_images.append(image)
    train_labels.append('seborrheic_keratosis')

print("[INFO] saving npz for training set...")
np.savez('data/training_data.npz', np.array(train_images))
np.savez('data/training_labels.npz', np.array(train_labels))

print("[INFO] preparaing test set")

for filename in os.listdir(melanoma_test_dir):
    image = cv.imread(base_path+melanoma_test_dir+filename)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = cv.resize(image, (100, 100), interpolation=cv.INTER_AREA)
    test_images.append(image)
    test_labels.append('melanoma')
    
for filename in os.listdir(nevus_test_dir):
    image = cv.imread(base_path+nevus_test_dir+filename)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = cv.resize(image, (100, 100), interpolation=cv.INTER_AREA)
    test_images.append(image)
    test_labels.append('nevus')
    
for filename in os.listdir(seborrheic_keratosis_test_dir):
    image = cv.imread(base_path+seborrheic_keratosis_test_dir+filename)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = cv.resize(image, (100, 100), interpolation=cv.INTER_AREA)
    test_images.append(image)
    test_labels.append('seborrheic_keratosis')

print("[INFO] saving npz for test set...")
np.savez('data/test_data.npz', np.array(train_images))
np.savez('data/test_labels.npz', np.array(train_labels))

print("[INFO] completed!")