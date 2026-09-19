"""
Train a convolutional neural network on MNIST.

Run from the project root:
    py -m examples.cnn_mnist

If this dosn't work, try:
    python -m examples.cnn_mnist

If CNN takes suspiciously long per epoch, change mode="epoch" to mode="batch" within ProgressBar.
"""

import tensorflow as tf

from nnframework.model import NeuralNetwork
from nnframework.layers import Conv2D, MaxPool2D, Flatten, Dense
from nnframework.activations import ReLU, Softmax
from nnframework.losses import CrossEntropyLoss
from nnframework.optimisers import Adam
from nnframework.metrics import CategoricalAccuracy
from nnframework.callbacks import EarlyStopping, ModelCheckpoint, ProgressBar


# MNIST contains 28 x 28 greyscale images of handwritten digits.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


model = NeuralNetwork(
    layers=[
        # One input greyscale channel becomes eight 26 x 26 feature maps.
        Conv2D(1, 8, 3, 1, 0, initialisation="he_normal"),
        ReLU(),

        # Decrease feature maps from 26 x 26 to 13 x 13.
        MaxPool2D(2, 2),

        # Learn more complex features.
        Conv2D(8, 16, 3, 1, 0, initialisation="he_normal"),
        ReLU(),

        # Decrease feature maps from 11 x 11 to 5 x 5.
        MaxPool2D(2, 2),

        # Convert 16 feature maps of size 5 x 5 into 400 values.
        Flatten(),

        Dense(16 * 5 * 5, 128, initialisation="he_normal"),
        ReLU(),

        # Produce values, interpreted as probabilites, for the ten possible digits.
        Dense(128, 10, initialisation="xavier_normal"),
        Softmax(),
    ],
    batch_size=64,
    Loss_Algorithm=CrossEntropyLoss(),
    Optimiser=Adam(learning_rate=0.001, beta1=0.9, beta2=0.999),
    Metrics=(CategoricalAccuracy(),),
)


model.train(
    x_train,
    y_train,
    epochs=12,
    normalise=True,
    validation_data=(x_test, y_test),
    CallBacks=[
        ProgressBar(mode="epoch", bar_length=50), # Convert epoch to batch if takes too long!
        EarlyStopping(monitor="val_loss", patience=3),
        ModelCheckpoint("best_cnn_mnist_model.npz"),
    ],
)

# Evaluate the final CNN on unseen test images.
model.test(x_test, y_test)