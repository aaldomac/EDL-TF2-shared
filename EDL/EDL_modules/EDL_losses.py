# Import necessary libraries
import numpy as np
np.typeDict = np.sctypeDict
import tensorflow as tf


class EDLoss_PAC(tf.keras.losses.Loss):
    """
    This class applies Eq. 5 in the paper. This equation corresponds to the sum of squares loss plus a variance term and
    a regularizer term. The regularizer term is the KL divergence between the predicted Dirichlet distribution and a
    uniform Dirichlet distribution.
    """
    def __init__(self, logits2evidence: tf.keras.layers.Activation, annealing_step=10, name='EDLoss_PAC'):
        super().__init__(name=name, reduction=tf.keras.losses.Reduction.AUTO)
        self.logits2evidence = logits2evidence
        self.annealing_step = annealing_step
        self._current_epoch = 0

    @staticmethod
    def KL(alpha_tilda):
        beta = tf.ones_like(alpha_tilda)  # uniform distribution with which we aim to compare
        strength_alpha = tf.reduce_sum(alpha_tilda, axis=1, keepdims=True)
        strength_beta = tf.reduce_sum(beta, axis=1, keepdims=True)
        log_gamma_alpha = (tf.math.lgamma(strength_alpha)
                           - tf.reduce_sum(tf.math.lgamma(alpha_tilda), axis=1, keepdims=True))
        log_gamma_beta = (tf.reduce_sum(tf.math.lgamma(beta), axis=1, keepdims=True)
                          - tf.math.lgamma(strength_beta))  # = 0 - lgamma(10)

        dg0 = tf.math.digamma(strength_alpha)  # digamma function of the strength of our distribution
        dg1 = tf.math.digamma(alpha_tilda)  # digamma function of the Dirichlet parameters individually

        kl = tf.reduce_sum((alpha_tilda - beta) * (dg1 - dg0), axis=1, keepdims=True) + log_gamma_alpha + log_gamma_beta
        return kl

    def call(self, y_true, logits, model=None):
        if self.logits2evidence.name == "exponential":
            print(f"Name of activation function: {self.logits2evidence.name}")
            evidence = self.logits2evidence(tf.clip_by_value(logits, -10, 10))

        else:
            evidence = self.logits2evidence(logits)
        alpha = evidence + 1
        strength = tf.reduce_sum(alpha, axis=1, keepdims=True)
        y_pred = alpha / strength  # predicted probabilities

        mse_term = tf.reduce_sum((y_true - y_pred) ** 2, axis=1, keepdims=True)  # mse loss term
        var_term = tf.reduce_sum(alpha * (strength - alpha) / (strength * strength * (strength + 1)), axis=1,
                                 keepdims=True)  # variance term
        # Equivalent form for the variance term
        # var_term = tf.reduce_sum(y_pred * (1 - y_pred) / (strength + 1), axis=1, keepdims=True)

        annealing_coef = tf.minimum(1.0, tf.cast(self._current_epoch / self.annealing_step, tf.float32))  # depends on the current step

        # With the next line we only get the misleading alphas
        alpha_tilda = evidence * (1 - y_true) + 1  # dirichlet parameters
        c = annealing_coef * self.KL(alpha_tilda)  # regularize the KL to not converge fast to uniform distribution

        loss = mse_term + var_term + c  # error + variance + regularizer

        return tf.reduce_mean(loss)

    def set_epoch(self, value):
        self._current_epoch = value


class EDLoss_Gibbs(tf.keras.losses.Loss):
    """
    This class applies Eq. 3 and Eq. 4 in the paper. Depending on the function used we have Eq. 3 (function=log) for
    Bayes classifier or Eq. 4 (function=digamma) for a Gibbs classifier.
    """
    def __init__(self, function, logits2evidence: tf.keras.layers.Activation,
                 annealing_step=10, name='EDLoss_Gibbs'):
        super().__init__(name=name)
        self.function = function
        self.logits2evidence = logits2evidence
        self.annealing_step = annealing_step
        self._current_epoch = 0

    @staticmethod
    def KL(alpha):
        beta = tf.ones_like(alpha)  # uniform distribution with which we aim to compare
        strength_alpha = tf.reduce_sum(alpha, axis=1, keepdims=True)
        strength_beta = tf.reduce_sum(beta, axis=1, keepdims=True)
        log_gamma_alpha = tf.math.lgamma(strength_alpha) - tf.reduce_sum(tf.math.lgamma(alpha), axis=1, keepdims=True)
        log_gamma_beta = (tf.reduce_sum(tf.math.lgamma(beta), axis=1, keepdims=True)
                          - tf.math.lgamma(strength_beta))  # = 0

        dg0 = tf.math.digamma(strength_alpha)  # digamma function of the strength of our distribution
        dg1 = tf.math.digamma(alpha)  # digamma function of the Dirichlet parameters individually

        kl = tf.reduce_sum((alpha - beta) * (dg1 - dg0), axis=1, keepdims=True) + log_gamma_alpha + log_gamma_beta
        return kl

    def call(self, y_true, logits, epoch=10, model=None):
        if self.logits2evidence.name == "exponential":
            print(f"Name of activation function: {self.logits2evidence.name}")
            logits = tf.clip_by_value(logits, -10, 10)

        evidence = self.logits2evidence(logits)
        alpha = evidence + 1
        strength = tf.reduce_sum(alpha, axis=1, keepdims=True)

        # Gibbs sampling
        a = tf.reduce_sum(y_true * (self.function(strength) - self.function(alpha)), axis=1, keepdims=True)

        annealing_coef = tf.minimum(1.0, tf.cast(epoch / self.annealing_step, tf.float32))  # depends on the current step

        # With the next line we only get the misleading alphas
        alpha_tilda = evidence * (1 - y_true) + 1  # dirichlet parameters
        c = annealing_coef * self.KL(alpha_tilda)  # regularize the KL to not converge fast to uniform distribution

        loss = tf.reduce_mean(a + c)  # error + regularizer

        return loss

    def set_epoch(self, value):
        self._current_epoch = value
