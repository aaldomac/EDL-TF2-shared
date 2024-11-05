# Import necessary libraries
import numpy as np

np.typeDict = np.sctypeDict
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import scipy.ndimage as nd


class LeNet_softmax(tf.keras.Model):
    """
    LeNet model with softmax output layer.
    """
    def __init__(self, regularizer_term=0.005, name='LeNet_softmax'):
        super(LeNet_softmax, self).__init__()

        self.regularizer_term = regularizer_term
        # self.name = name

        # Add first convolutional block
        self.conv1 = tf.keras.layers.Conv2D(20, kernel_size=(5, 5),
                                            kernel_regularizer=tf.keras.regularizers.L2(self.regularizer_term),
                                            strides=(1, 1), activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        # Add second convolutional block
        self.conv2 = tf.keras.layers.Conv2D(50, kernel_size=(5, 5),
                                            kernel_regularizer=tf.keras.regularizers.L2(self.regularizer_term),
                                            strides=(1, 1), activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        # Flatten the output
        self.flatten = tf.keras.layers.Flatten()

        # Add fully connected layer
        self.dense_1 = tf.keras.layers.Dense(500,
                                             kernel_regularizer=tf.keras.regularizers.L2(self.regularizer_term),
                                             activation='relu')
        self.dropout = tf.keras.layers.Dropout(rate=0.5)

        # Output layer
        self.dense_2 = tf.keras.layers.Dense(10,
                                             kernel_regularizer=tf.keras.regularizers.L2(self.regularizer_term))

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.dense_1(x)

        if training:
            x = self.dropout(x, training=training)

        logits = self.dense_2(x)

        return logits


class LeNet_EDL(tf.keras.Model):
    """
    LeNet model with Evidential Deep Learning (EDL) approach.
    """
    def __init__(self, regularizer_term=0.005, K=10, name='LeNet_EDL'):
        super(LeNet_EDL, self).__init__()

        self.K = K
        self.regularizer_term = regularizer_term

        # Add first convolutional block
        self.conv1 = tf.keras.layers.Conv2D(20, kernel_size=(5, 5),
                                            kernel_regularizer=tf.keras.regularizers.L2(self.regularizer_term),
                                            strides=(1, 1), activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        # Add second convolutional block
        self.conv2 = tf.keras.layers.Conv2D(50, kernel_size=(5, 5),
                                            kernel_regularizer=tf.keras.regularizers.L2(self.regularizer_term),
                                            strides=(1, 1), activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        # Flatten the output
        self.flatten = tf.keras.layers.Flatten()

        # Add fully connected layer
        self.dense_1 = tf.keras.layers.Dense(500,
                                             kernel_regularizer=tf.keras.regularizers.L2(self.regularizer_term),
                                             activation='relu')
        self.dropout = tf.keras.layers.Dropout(rate=0.5)

        # Output layer
        self.dense_2 = tf.keras.layers.Dense(10, activation='linear',
                                             kernel_regularizer=tf.keras.regularizers.L2(self.regularizer_term))

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.dense_1(x)

        if training:
            x = self.dropout(x, training=training)

        logits = self.dense_2(x)

        return logits

    def predict(self, inputs, uncertainty=False, **kwargs):
        """
        Predict the output of the model.
        """
        logits = self(inputs)
        if uncertainty:
            evidence = tf.keras.layers.Activation('relu')(logits)
            alpha = evidence + 1

            uncertainty_estimation = self.K / tf.reduce_sum(alpha, axis=1, keepdims=False)
            return uncertainty_estimation

        else:
            return logits
