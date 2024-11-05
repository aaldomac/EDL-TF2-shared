# Import necessary libraries

import numpy as np
np.typeDict = np.sctypeDict
import tensorflow as tf
import matplotlib.pyplot as plt

from EDL.EDL_modules.EDL_models import LeNet_softmax, LeNet_EDL
from EDL.EDL_modules.EDL_utils import load_mnist, rotating_image_classification, draw_EDL_results, mixing_digits
from EDL.EDL_modules.EDL_losses import EDLoss_PAC, EDLoss_Gibbs
from EDL.EDL_modules.EDL_metrics import EDL_Accuracy, EDL_mean_ev, EDL_mean_ev_succ, EDL_mean_ev_fail
from EDL.EDL_modules.EDL_callbacks import UpdateEpochCallback

# Enable eager execution
# tf.config.run_functions_eagerly(True)

TEST_LENET = False
TEST_EQ3 = False
TEST_EQ4 = False
TEST_EQ5 = True
DIGIT_MIX = False

if __name__ == '__main__':
    # Import MNIST dataset
    (x_train, y_train), (x_test, y_test), num_classes = load_mnist()

    # Define the batch_size and the number of batches
    batch_size = 1000
    num_batches = x_train.shape[0] // batch_size
    print(f"Number of batches: {num_batches}, x_train shape: {x_train.shape}")

    # Select an image from the dataset corresponding to digit 1 and plot it
    digit_one = x_train[np.argmax(y_train, axis=1) == 1][0]
    fig, ax = plt.subplots()
    ax.imshow(digit_one)
    ax.set_title("Digit 1 example"), ax.axis('off')


    if TEST_LENET:
        # Train the LeNet network
        LeNet_softmax_model = LeNet_softmax()
        LeNet_softmax_model.compile(optimizer=tf.keras.optimizers.Adam(),
                                    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                                    metrics=['accuracy'])
        LeNet_softmax_model.fit(x_train, y_train, epochs=10, batch_size=256)

        # See how the predictions change as the image rotates
        rotating_image_classification(digit_one, LeNet_softmax_model)

        # Mix digits experiment
        if DIGIT_MIX:
            mixing_digits(LeNet_softmax_model, x_train, y_train)

    if TEST_EQ3:
        # Define the other loss function (Eq. 3 of the paper)
        logits2evidence = tf.keras.layers.Activation("exponential", name="exponential")
        loss_fn = EDLoss_Gibbs(function=tf.math.log, logits2evidence=logits2evidence)
        metrics = [EDL_Accuracy(), EDL_mean_ev(logits2evidence), EDL_mean_ev_succ(logits2evidence),
                   EDL_mean_ev_fail(logits2evidence)]

        # Create the LeNet_EDL network, compile and train
        LeNet_EDL_model = LeNet_EDL()
        LeNet_EDL_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss_fn, metrics=metrics)
        history = LeNet_EDL_model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test),
                                      batch_size=batch_size)

        # Plot the training history
        draw_EDL_results(history)

        # Plot results on rotating image
        rotating_image_classification(digit_one, LeNet_EDL_model, uncertainty=True)

        # Mix digits experiment
        if DIGIT_MIX:
            mixing_digits(LeNet_EDL_model, x_train, y_train, uncertainty=True)

    if TEST_EQ4:
        # Define the other loss function (Eq. 4 of the paper)
        logits2evidence = tf.keras.layers.Activation("exponential", name="exponential")
        loss_fn = EDLoss_Gibbs(function=tf.math.digamma, logits2evidence=logits2evidence)
        metrics = [EDL_Accuracy(), EDL_mean_ev(logits2evidence), EDL_mean_ev_succ(logits2evidence),
                   EDL_mean_ev_fail(logits2evidence)]

        # Create the LeNet_EDL network, compile and train
        LeNet_EDL_model = LeNet_EDL()
        LeNet_EDL_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss_fn, metrics=metrics)
        history = LeNet_EDL_model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test),
                                      batch_size=batch_size)

        # Plot the training history
        draw_EDL_results(history)

        # Plot results on rotating image
        rotating_image_classification(digit_one, LeNet_EDL_model, uncertainty=True)

        # Mix digits experiment
        if DIGIT_MIX:
            mixing_digits(LeNet_EDL_model, x_train, y_train, uncertainty=True)

    if TEST_EQ5:
        # Create the LeNet_EDL network, compile and train
        logits2evidence = tf.keras.layers.Activation("relu")
        loss_fn = EDLoss_PAC(logits2evidence=logits2evidence)
        metrics = [EDL_Accuracy(), EDL_mean_ev(logits2evidence), EDL_mean_ev_succ(logits2evidence),
                   EDL_mean_ev_fail(logits2evidence)]
        LeNet_EDL_model = LeNet_EDL()
        LeNet_EDL_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss_fn, metrics=metrics)
        history = LeNet_EDL_model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test),
                                      batch_size=batch_size, callbacks=[UpdateEpochCallback(loss_fn)])

        # Plot the training history
        draw_EDL_results(history)

        # Plot results on rotating image
        rotating_image_classification(digit_one, LeNet_EDL_model, uncertainty=True)

        # Mix digits experiment
        if DIGIT_MIX:
            mixing_digits(LeNet_EDL_model, x_train, y_train, uncertainty=True)

    plt.show()
