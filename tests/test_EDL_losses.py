from unittest import TestCase
import numpy as np
np.typeDict = np.sctypeDict
import tensorflow as tf

from EDL.EDL_modules.EDL_losses import EDLoss_PAC, EDLoss_Gibbs


class EDLoss_PAC_Test(TestCase):

    def test_EDLoss_PAC_init(self):
        logits2evidence = tf.keras.layers.Activation("relu")
        loss_fn = EDLoss_PAC(logits2evidence=logits2evidence)
        self.assertEqual(loss_fn.name, 'EDLoss_PAC')
        self.assertEqual(loss_fn.logits2evidence, logits2evidence)
        self.assertEqual(loss_fn.annealing_step, 10)
        self.assertEqual(loss_fn._current_epoch, 0)

    def test_EDLoss_KL(self):
        logits2evidence = tf.keras.layers.Activation("relu")
        loss_fn = EDLoss_PAC(logits2evidence=logits2evidence)
        alpha_tilda = tf.constant([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        kl = loss_fn.KL(alpha_tilda)
        self.assertEqual(kl.shape, (2, 1))
        self.assertTrue(np.allclose(kl.numpy(), [[0.0], [0.0]]))

    def test_EDLoss_set_epoch(self):
        logits2evidence = tf.keras.layers.Activation("relu")
        loss_fn = EDLoss_PAC(logits2evidence=logits2evidence)
        loss_fn.set_epoch(1)
        self.assertEqual(loss_fn._current_epoch, 1)

    def test_EDLoss_call_perfect(self):
        logits2evidence = tf.keras.layers.Activation("relu")
        loss_fn = EDLoss_PAC(logits2evidence=logits2evidence)
        y_true = tf.constant([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        logits = tf.constant([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        loss = loss_fn(y_true, logits)
        self.assertEqual(loss.shape, ())
        self.assertEqual(loss, 0.5)

    def test_EDLoss_call_wrong(self):
        logits2evidence = tf.keras.layers.Activation("relu")
        loss_fn = EDLoss_PAC(logits2evidence=logits2evidence)
        y_true = tf.constant([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        logits = tf.constant([[-1.0, 1.0, -1.0], [1.0, 0.0, 0.0]])  # negative logits do not affect the loss
        loss = loss_fn(y_true, logits)
        self.assertEqual(loss.shape, ())
        self.assertEqual(loss, 1.0)


class EDLoss_Gibbs_Test(TestCase):

    def test_EDLoss_Gibbs_init(self):
        logits2evidence = np.exp
        loss_fn = EDLoss_Gibbs(function=np.log, logits2evidence=logits2evidence)
        self.assertEqual(loss_fn.name, 'EDLoss_Gibbs')
        self.assertEqual(loss_fn.logits2evidence, logits2evidence)
        self.assertEqual(loss_fn.annealing_step, 10)
        self.assertEqual(loss_fn._current_epoch, 0)