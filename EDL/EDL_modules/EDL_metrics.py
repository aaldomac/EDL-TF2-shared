# Import necessary libraries
import numpy as np

np.typeDict = np.sctypeDict
import tensorflow as tf


class EDL_Accuracy(tf.keras.metrics.Metric):
    def __init__(self, name='EDL_Accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.accuracy = self.add_weight(name='accuracy', initializer='zeros')
        self.total_matches = self.add_weight(name='total_matches', initializer='zeros')

    def update_state(self, y_true, y_pred, **kwargs):
        logits = y_pred
        pred = tf.argmax(logits, axis=1)
        truth = tf.argmax(y_true, axis=1)
        match = tf.cast(tf.equal(pred, truth), tf.float32)

        # Update accuracy
        self.accuracy.assign_add(tf.reduce_sum(match))  # shape=()

        # Update match count for averaging
        self.total_matches.assign_add(tf.cast(tf.size(truth), tf.float32))

    def result(self):
        return self.accuracy / self.total_matches

    def reset_states(self):
        self.accuracy.assign(0)
        self.total_matches.assign(0)


class EDL_mean_ev(tf.keras.metrics.Metric):
    def __init__(self, logits2evidence: tf.keras.layers.Activation, name='EDL_mean_ev', **kwargs):
        super().__init__(name=name, **kwargs)
        self.logits2evidence = logits2evidence
        self.evidence = self.add_weight(name='evidence', initializer='zeros')
        self.total_matches = self.add_weight(name='total_matches', initializer='zeros')

    def update_state(self, y_true, y_pred, **kwargs):
        logits = y_pred  # Including the [0] would make it only take one array of the one of the three matrices
        if self.logits2evidence.name == "exponential":
            print(f"Name of activation function: {self.logits2evidence.name}")
            evidence = self.logits2evidence(tf.clip_by_value(logits, -10, 10))

        else:
            evidence = self.logits2evidence(logits)

        # Update evidence metrics
        total_evidence = tf.reduce_sum(evidence, axis=1, keepdims=True)
        self.evidence.assign_add(tf.reduce_sum(total_evidence))  # shape=()

        # Update match count for averaging
        self.total_matches.assign_add(tf.cast(tf.size(total_evidence), tf.float32))

    def result(self):
        return self.evidence / self.total_matches

    def reset_states(self):
        self.evidence.assign(0)
        self.total_matches.assign(0)


class EDL_mean_ev_succ(tf.keras.metrics.Metric):
    def __init__(self, logits2evidence: tf.keras.layers.Activation, name='EDL_mean_ev_succ', **kwargs):
        super().__init__(name=name, **kwargs)
        self.logits2evidence = logits2evidence
        self.ev_succ = self.add_weight(name='ev_succ', initializer='zeros')
        self.total_matches = self.add_weight(name='total_matches', initializer='zeros')

    def update_state(self, y_true, y_pred, **kwargs):
        logits = y_pred  # Including the [0] would make it only take one array of the one of the three matrices
        evidence = self.logits2evidence(logits)

        # Compute matches
        pred = tf.argmax(logits, axis=1)
        truth = tf.argmax(y_true, axis=1)
        match = tf.expand_dims(tf.cast(tf.equal(pred, truth), tf.float32), axis=-1)

        # Update evidence metrics
        total_evidence = tf.reduce_sum(evidence, axis=1, keepdims=True)
        self.ev_succ.assign_add(tf.reduce_sum(total_evidence * match))  #shape=()

        # Update match count for averaging
        self.total_matches.assign_add(tf.reduce_sum(match + 1e-20))

    def result(self):
        return self.ev_succ / self.total_matches

    def reset_states(self):
        self.ev_succ.assign(0)
        self.total_matches.assign(0)


class EDL_mean_ev_fail(tf.keras.metrics.Metric):
    def __init__(self, logits2evidence: tf.keras.layers.Activation, name='EDL_mean_ev_fail', **kwargs):
        super().__init__(name=name, **kwargs)
        self.logits2evidence = logits2evidence
        self.ev_fail = self.add_weight(name='ev_fail', initializer='zeros')
        self.total_mismatches = self.add_weight(name='total_mismatches', initializer='zeros')

    def update_state(self, y_true, y_pred, **kwargs):
        logits = y_pred
        evidence = self.logits2evidence(logits)

        # Compute matches
        pred = tf.argmax(logits, axis=1)
        truth = tf.argmax(y_true, axis=1)
        match = tf.expand_dims(tf.cast(tf.equal(pred, truth), tf.float32), axis=-1)

        # Update evidence metrics
        total_evidence = tf.reduce_sum(evidence, axis=1, keepdims=True)
        self.ev_fail.assign_add(tf.reduce_sum(total_evidence * tf.abs(1 - match)))  # shape=()

        # Update match count for averaging
        self.total_mismatches.assign_add(tf.reduce_sum(tf.abs(1 - match) + 1e-20))

    def result(self):
        return self.ev_fail / self.total_mismatches

    def reset_states(self):
        self.ev_fail.assign(0)
        self.total_mismatches.assign(0)
