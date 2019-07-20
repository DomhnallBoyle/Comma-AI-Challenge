"""
    Filename: 
    Description:
    Author: Domhnall Boyle
    Maintained by: Domhnall Boyle
    Email: dboyle25@qub.ac.uk
    Python Version: 3.6
"""

from keras.callbacks import Callback


class BatchCallback(Callback):

    def __init__(self):
        super().__init__()
        self.training_sample_index = 0
        self.validation_sample_index = 0
        self.begin_epoch = True

    def on_epoch_end(self, epoch, logs=None):
        self.training_sample_index = 0
        self.validation_sample_index = 0
        self.begin_epoch = True
