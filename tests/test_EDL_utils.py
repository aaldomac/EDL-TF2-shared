from unittest import TestCase
import numpy as np
np.typeDict = np.sctypeDict
from EDL.EDL_modules.EDL_utils import load_mnist, load_cifar10, rotating_image_classification, mixing_digits, rotate_img


class TestEDLUtils(TestCase):

    def test_load_mnist(self):
        (x_train, y_train), (x_test, y_test), num_classes = load_mnist()
        self.assertEqual(x_train.shape, (60000, 28, 28, 1))
        self.assertEqual(y_train.shape, (60000, 10))
        self.assertEqual(x_test.shape, (10000, 28, 28, 1))
        self.assertEqual(y_test.shape, (10000, 10))
        self.assertEqual(num_classes, 10)

    def test_load_cifar10(self):
        (x_train, y_train), (x_test, y_test), num_classes = load_cifar10()
        self.assertEqual(x_train.shape, (50000, 32, 32, 3))
        self.assertEqual(y_train.shape, (50000, 10))
        self.assertEqual(x_test.shape, (10000, 32, 32, 3))
        self.assertEqual(y_test.shape, (10000, 10))
        self.assertEqual(num_classes, 10)

    def test_rotate_img(self):
        x = np.random.rand(28, 28)
        deg = 45
        rotated_img = rotate_img(x, deg)
        self.assertEqual(rotated_img.shape, (784,))
