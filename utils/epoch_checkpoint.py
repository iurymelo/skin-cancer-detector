# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 00:49:56 2020

@author: iurym
"""

from keras.callbacks import Callback
import os

class EpochCheckpoint(Callback):
    def __init__(self, outputPath, every=10, startAt=0):
        # Call the parent class constructor
        super(Callback, self).__init__()
        
        # store the base output path for the model, the number of
		# epochs that must pass before the model is serialized to
		# disk and the current epoch value
        self.outputPath = outputPath
        self.every = every
        self.intEpoch = startAt
        
    def on_epoch_end(self, epoch, logs={}):
        # Check to see if model should be serialized to disk
        if (self.intEpoch + 1) % self.every == 0:
            path = os.path.sep.join([self.outputPath,
                                     "epoch_{}.hdf5".format(self.intEpoch + 1)])
            self.model.save(path, overwrite=True)
        
        self.intEpoch += 1