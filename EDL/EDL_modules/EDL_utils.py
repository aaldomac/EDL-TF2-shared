# Import necessary libraries
import numpy as np
np.typeDict = np.sctypeDict
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import scipy.ndimage as nd
from scipy.special import softmax


# Function for downloading the data
def load_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = np.expand_dims(x_train, -1), np.expand_dims(x_test, -1)
    x_train, x_test = x_train / 255, x_test / 255
    y_train, y_test = tf.keras.utils.to_categorical(y_train), tf.keras.utils.to_categorical(y_test)

    num_classes = y_train.shape[1]

    return (x_train, y_train), (x_test, y_test), num_classes


# Create the rotate image function
def rotate_img(x, deg):
    return nd.rotate(x, deg, reshape=False).ravel()


# This method rotates an image counter-clockwise and classify it for different degress of rotation.
# It plots the highest classification probability along with the class label for each rotation degree.
def rotating_image_classification(img, model, uncertainty=False, threshold=0.5):
    max_deg = 180
    step_deg = int(max_deg / 10) + 1
    deg_array = []
    prob_array = []
    lu = []
    K = 10 #model.output_shape[-1]  # Number of classes, should be 10 for MNIST
    scores = np.zeros((1, K))
    rimgs = np.zeros((28, 28, step_deg)) # [28, 28, 19]

    for i, deg in enumerate(np.linspace(0, max_deg, step_deg)):
        step_img = rotate_img(img, deg).reshape(28, 28)
        step_img = np.clip(a=step_img, a_min=0, a_max=1)
        rimgs[:, :, i] = step_img

        # Prepare input for the model
        step_img_expanded = np.expand_dims(step_img, axis=(0, -1))  # Adding batch and channel dimensions

        if uncertainty:
            logits_pred_t = model.predict(step_img_expanded)
            # p_pred_t_from_softmax = softmax(logits_pred_t)
            p_pred_t = tf.keras.layers.Activation('relu')(logits_pred_t)
            p_pred_t = (p_pred_t + 1) / tf.reduce_sum(p_pred_t + 1, axis=1, keepdims=True)
            uncertainty = model.predict(step_img_expanded, uncertainty=True)
            lu.append(np.mean(uncertainty))

        else:
            logits_pred_t = model.predict(step_img_expanded)
            # p_pred_t_from_softmax = softmax(logits_pred_t)
            p_pred_t = tf.keras.layers.Activation('relu')(logits_pred_t)
            p_pred_t = (p_pred_t + 1) / tf.reduce_sum(p_pred_t + 1, axis=1, keepdims=True)

        scores += tf.cast(p_pred_t >= threshold, tf.int32)
        # print(f"p_pred_t {p_pred_t}")
        deg_array.append(deg)
        prob_array.append(p_pred_t[0])
    labels = np.arange(10)[tf.cast(scores[0], tf.bool)]
    prob_array = np.array(prob_array)[:, labels]
    c = ['black', 'blue', 'red', 'brown', 'purple', 'cyan']
    marker = ['s', '^', 'o'] * 2
    labels = labels.tolist()

    fig, ax = plt.subplots()
    for i in range(len(labels)):
        ax.plot(deg_array, prob_array[:, i], marker=marker[i], c=c[i])

    if uncertainty:
        labels += ['uncertainty']
        ax.plot(deg_array, lu, marker='<', c='red')

    plt.legend(labels)
    ax.set_xlim([0, max_deg])
    ax.set_xlabel('Rotation Degree')
    ax.set_ylabel('Classification Probability')

    ax.set_xticks(deg_array)
    # Create an OffsetImage and add it to each xtick
    for i, angle in enumerate(deg_array):
        # Create OffsetImage object
        img = OffsetImage(1 - rimgs[:, :, i], cmap="gray", zoom=1)  # Adjust zoom for image size

        # Create AnnotationBbox with the image at the corresponding x-position and just below the x-axis
        ab = AnnotationBbox(img, (angle, 0.),  # Position slightly below the axis
                            xybox=(0, -60), frameon=False,  # xybox moves the image down
                            xycoords='data', boxcoords="offset points", pad=0)

        ax.add_artist(ab)

    plt.tight_layout()


def draw_EDL_results(history, K=10):
    # Extract the necessary data from the history
    train_acc1 = history.history['EDL_Accuracy']
    test_acc1 = history.history['val_EDL_Accuracy']
    train_ev_succ = history.history['EDL_mean_ev_succ']
    train_ev_fail = history.history['EDL_mean_ev_fail']
    test_ev_succ = history.history['val_EDL_mean_ev_succ']
    test_ev_fail = history.history['val_EDL_mean_ev_fail']

    # calculate uncertainty for training and testing data for correctly and misclassified samples
    train_u_succ = K / (K + np.array(train_ev_succ))
    train_u_fail = K / (K + np.array(train_ev_fail))
    test_u_succ = K / (K + np.array(test_ev_succ))
    test_u_fail = K / (K + np.array(test_ev_fail))

    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches([10, 10])

    axs[0, 0].plot(train_ev_succ, c='r', marker='+')
    axs[0, 0].plot(train_ev_fail, c='k', marker='x')
    axs[0, 0].set_title('Train Data')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Estimated total evidence for classification')
    axs[0, 0].legend(['Correct Classifications', 'Misclassifications'])

    axs[0, 1].plot(train_u_succ, c='r', marker='+')
    axs[0, 1].plot(train_u_fail, c='k', marker='x')
    axs[0, 1].plot(train_acc1, c='blue', marker='*')
    axs[0, 1].set_title('Train Data')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Estimated uncertainty for classification')
    axs[0, 1].legend(['Correct classifications', 'Misclassifications', 'Accuracy'])

    axs[1, 0].plot(test_ev_succ, c='r', marker='+')
    axs[1, 0].plot(test_ev_fail, c='k', marker='x')
    axs[1, 0].set_title('Test Data')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Estimated total evidence for classification')
    axs[1, 0].legend(['Correct Classifications', 'Misclassifications'])

    axs[1, 1].plot(test_u_succ, c='r', marker='+')
    axs[1, 1].plot(test_u_fail, c='k', marker='x')
    axs[1, 1].plot(test_acc1, c='blue', marker='*')
    axs[1, 1].set_title('Test Data')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Estimated uncertainty for classification')
    axs[1, 1].legend(['Correct classifications', 'Misclassifications', 'Accuracy'])


def mixing_digits(model, x_train, y_train, uncertainty=None):
    image_0 = x_train[np.argmax(y_train, axis=1) == 0][0]
    image_6 = x_train[np.argmax(y_train, axis=1) == 6][0]
    image_mix = 0.5 * image_0 + 0.5 * image_6
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(image_0), ax[0].set_title("Digit 0")
    ax[1].imshow(image_6), ax[1].set_title("Digit 6")
    ax[2].imshow(image_mix), ax[2].set_title("Digit mix")

    logits2evidence = tf.keras.layers.Activation('relu')

    # Predict the mix of digits
    prediction = model.predict(image_mix[np.newaxis, ...])
    prediction_evidence = logits2evidence(prediction)
    prediction_alpha = prediction_evidence + 1
    prediction_probability = prediction_alpha / tf.reduce_sum(prediction_alpha, axis=1, keepdims=True)
    print(f"{model.name} prediction probability for the mix of digits: {prediction_probability}")

    if uncertainty is not None:
        # Give the uncertainty of the prediction
        uncertainty = model.predict(image_mix[np.newaxis, ...], uncertainty=True)
        print(f"{model.name} prediction uncertainty for the mix of digits: {uncertainty}")