from unittest import TestCase
import numpy as np
np.typeDict = np.sctypeDict
import tensorflow as tf

from EDL.EDL_modules.EDL_metrics import EDL_Accuracy, EDL_mean_ev, EDL_mean_ev_succ, EDL_mean_ev_fail


class EDL_Accuracy_MetricsTest(TestCase):

    def test_EDL_Accuracy_init(self):
        metric = EDL_Accuracy()
        self.assertEqual(metric.name, 'EDL_Accuracy')
        self.assertEqual(metric.accuracy, 0)
        self.assertEqual(metric.total_matches, 0)

    def test_EDL_Accuracy_update_state_and_reset(self):
        metric = EDL_Accuracy()
        y_true = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
        y_pred = [[7.0, 0.0, 0.0], [0.0, 9.0, 0.0]]  # y_pred are Dirichlet parameters
        metric.update_state(y_true, y_pred)
        self.assertEqual(metric.accuracy, 0.0)
        self.assertEqual(metric.total_matches, 2.0)  # batch size is 2
        self.assertEqual(metric.result(), 0.0)
        metric.reset_states()
        self.assertEqual(metric.accuracy, 0)
        self.assertEqual(metric.total_matches, 0)

    def test_EDL_Accuracy_update_state_twice(self):
        metric = EDL_Accuracy()
        y_true = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
        y_pred = [[0.0, 2.0, 0.0], [5.0, 0.0, 0.0]]  # y_pred are Dirichlet parameters
        metric.update_state(y_true, y_pred)
        self.assertEqual(metric.accuracy, 2.0)  # max in Dirichlet parameters coincide with max in one-hot
        self.assertEqual(metric.total_matches, 2.0)  # batch size is 2
        self.assertEqual(metric.result(), 1.0)
        y_pred = [[0.0, 0.5, 0.0], [0.0, 0.1, 0.0]]  # y_pred are Dirichlet parameters
        metric.update_state(y_true, y_pred)
        self.assertEqual(metric.accuracy, 3.0)
        self.assertEqual(metric.total_matches, 4.0)  # batch size is 2 but we call update_state twice
        self.assertEqual(metric.result(), 0.75)


class EDL_Mean_Evidence_MetricsTest(TestCase):

    def test_EDL_mean_ev_init(self):
        logits2evidence = tf.keras.layers.Activation("relu")
        metric = EDL_mean_ev(logits2evidence)
        self.assertEqual(metric.name, 'EDL_mean_ev')
        self.assertEqual(metric.logits2evidence, logits2evidence)
        self.assertEqual(metric.evidence, 0)
        self.assertEqual(metric.total_matches, 0)

    def test_EDL_mean_ev_update_state_and_reset(self):
        logits2evidence = tf.keras.layers.Activation("relu")
        metric = EDL_mean_ev(logits2evidence)
        y_true = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
        y_pred = [[7.0, 0.0, 0.0], [0.0, 9.0, 0.0]]  # y_pred are Dirichlet parameters
        metric.update_state(y_true, y_pred)
        self.assertEqual(metric.evidence, 16.0)
        self.assertEqual(metric.total_matches, 2.0)
        self.assertEqual(metric.result(), 8.0)
        metric.reset_states()
        self.assertEqual(metric.evidence, 0)
        self.assertEqual(metric.total_matches, 0)

    def test_EDL_mean_ev_update_state_twice(self):
        logits2evidence = tf.keras.layers.Activation("relu")
        metric = EDL_mean_ev(logits2evidence)
        y_true = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
        y_pred = [[0.0, 2.0, 0.0], [5.0, 0.0, 0.0]]
        metric.update_state(y_true, y_pred)
        self.assertEqual(metric.evidence, 7.0)
        self.assertEqual(metric.total_matches, 2.0)
        self.assertEqual(metric.result(), 3.5)
        y_pred = [[0.0, 0.5, 0.0], [0.0, 0.1, 0.0]]
        metric.update_state(y_true, y_pred)
        self.assertEqual(metric.evidence, 7.6)
        self.assertEqual(metric.total_matches, 4.0)
        self.assertEqual(metric.result(), 1.9)


class EDL_Mean_Evidence_Success_MetricsTest(TestCase):

    def test_EDL_mean_ev_succ_init(self):
        logits2evidence = tf.keras.layers.Activation("relu")
        metric = EDL_mean_ev_succ(logits2evidence)
        self.assertEqual(metric.name, 'EDL_mean_ev_succ')
        self.assertEqual(metric.logits2evidence, logits2evidence)
        self.assertEqual(metric.ev_succ, 0)
        self.assertEqual(metric.total_matches, 0)

    def test_EDL_mean_ev_succ_update_state_and_reset(self):
        logits2evidence = tf.keras.layers.Activation("relu")
        metric = EDL_mean_ev_succ(logits2evidence)
        y_true = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
        y_pred = [[7.0, 0.0, 0.0], [9.0, 0.0, 0.0]]  # y_pred are Dirichlet parameters
        metric.update_state(y_true, y_pred)
        self.assertEqual(metric.ev_succ, 9.0)
        self.assertEqual(metric.total_matches, 1.0)
        self.assertEqual(metric.result(), 9.0)
        metric.reset_states()
        self.assertEqual(metric.ev_succ, 0)
        self.assertEqual(metric.total_matches, 0)

    def test_EDL_mean_ev_succ_update_state_twice(self):
        logits2evidence = tf.keras.layers.Activation("relu")
        metric = EDL_mean_ev_succ(logits2evidence)
        y_true = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
        y_pred = [[0.0, 2.0, 0.0], [5.0, 0.0, 0.0]]
        metric.update_state(y_true, y_pred)
        self.assertEqual(metric.ev_succ, 7.0)
        self.assertEqual(metric.total_matches, 2.0)
        self.assertEqual(metric.result(), 3.5)
        y_pred = [[0.0, 0.5, 0.0], [0.0, 0.1, 0.0]]
        metric.update_state(y_true, y_pred)
        self.assertEqual(metric.ev_succ, 7.5)
        self.assertEqual(metric.total_matches, 3.0)
        self.assertEqual(metric.result(), 2.5)


class EDL_Mean_Evidence_Fail_MetricsTest(TestCase):

    def test_EDL_mean_ev_fail_init(self):
        logits2evidence = tf.keras.layers.Activation("relu")
        metric = EDL_mean_ev_fail(logits2evidence)
        self.assertEqual(metric.name, 'EDL_mean_ev_fail')
        self.assertEqual(metric.logits2evidence, logits2evidence)
        self.assertEqual(metric.ev_fail, 0)
        self.assertEqual(metric.total_mismatches, 0)

    def test_EDL_mean_ev_fail_update_state_and_reset(self):
        logits2evidence = tf.keras.layers.Activation("relu")
        metric = EDL_mean_ev_fail(logits2evidence)
        y_true = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
        y_pred = [[7.0, 0.0, 0.0], [9.0, 0.0, 0.0]]  # y_pred are Dirichlet parameters
        metric.update_state(y_true, y_pred)
        self.assertEqual(metric.ev_fail, 7.0)
        self.assertEqual(metric.total_mismatches, 1.0)
        self.assertEqual(metric.result(), 7.0)
        metric.reset_states()
        self.assertEqual(metric.ev_fail, 0)
        self.assertEqual(metric.total_mismatches, 0)

    def test_EDL_mean_ev_fail_update_state_twice(self):
        logits2evidence = tf.keras.layers.Activation("relu")
        metric = EDL_mean_ev_fail(logits2evidence)
        y_true = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
        y_pred = [[0.0, 2.0, 0.0], [5.0, 0.0, 0.0]]
        metric.update_state(y_true, y_pred)
        self.assertEqual(metric.ev_fail, 0.0)
        self.assertEqual(metric.total_mismatches, 2e-20)
        self.assertEqual(metric.result(), 0.0)
        y_pred = [[0.0, 0.5, 0.0], [0.0, 0.1, 0.0]]
        metric.update_state(y_true, y_pred)
        self.assertEqual(metric.ev_fail, 0.1)
        self.assertEqual(metric.total_mismatches, 1.0)
        self.assertEqual(metric.result(), 0.1)
