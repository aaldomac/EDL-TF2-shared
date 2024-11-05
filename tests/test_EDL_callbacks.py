from unittest import TestCase
import numpy as np
np.typeDict = np.sctypeDict
import tensorflow as tf

from EDL.EDL_modules.EDL_callbacks import UpdateEpochCallback
from EDL.EDL_modules.EDL_losses import EDLoss_PAC


class EDLCallbacksTest(TestCase):

    def test_update_epoch_callback_init(self):
        loss_fn = EDLoss_PAC(logits2evidence=tf.keras.layers.Activation("relu"))
        callback = UpdateEpochCallback(loss_fn)
        self.assertEqual(callback.loss_fn, loss_fn)

    def test_update_epoch_callback_on_epoch_begin(self):
        loss_fn = EDLoss_PAC(logits2evidence=tf.keras.layers.Activation("relu"))
        callback = UpdateEpochCallback(loss_fn)
        callback.on_epoch_begin(epoch=1)
        self.assertEqual(loss_fn._current_epoch, 1)