from unittest import TestCase
import numpy as np
np.typeDict = np.sctypeDict
import tensorflow as tf

from EDL.EDL_modules.EDL_models import LeNet_EDL


class EDL_ModelsTest(TestCase):

    def test_LeNet_EDL_init(self):
        model = LeNet_EDL()
        self.assertEqual(model.name, 'LeNet_EDL')
        self.assertEqual(model.regularizer_term, 0.005)
        self.assertEqual(model.K, 10)
        self.assertEqual(len(model.layers), 8)
        self.assertEqual(model.layers[0].name, 'conv2d')
        self.assertEqual(model.layers[1].name, 'max_pooling2d')
        self.assertEqual(model.layers[2].name, 'conv2d_1')
        self.assertEqual(model.layers[3].name, 'max_pooling2d_1')
        self.assertEqual(model.layers[4].name, 'flatten')
        self.assertEqual(model.layers[5].name, 'dense')
        self.assertEqual(model.layers[6].name, 'dropout')
        self.assertEqual(model.layers[7].name, 'dense_1')