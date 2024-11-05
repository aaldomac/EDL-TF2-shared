# Import necessary libraries
import numpy as np
np.typeDict = np.sctypeDict
import tensorflow as tf


class UpdateEpochCallback(tf.keras.callbacks.Callback):
    def __init__(self, loss_fn):
        super().__init__()
        self.loss_fn = loss_fn

    def on_epoch_begin(self, epoch, logs=None):
        self.loss_fn.set_epoch(epoch)
