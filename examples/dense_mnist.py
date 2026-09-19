"""
Train a fully connected neural network on MNIST.

Run from the project root:
    py -m examples.dense_mnist
    
If this dosn't work, try:
    python -m examples.dense_mnist
"""

import tensorflow as tf

from nnframework.model import NeuralNetwork
from nnframework.layers import Dense
from nnframework.activations import ReLU, Softmax
from nnframework.losses import CrossEntropyLoss
from nnframework.optimisers import Adam
from nnframework.metrics import CategoricalAccuracy
from nnframework.callbacks import EarlyStopping, CSVLogger, ProgressBar


# MNIST contains 28 x 28 greyscale images of handwritten digits.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


model = NeuralNetwork(
    layers=[
        # The training loop flattens each 28 x 28 image into 784 pixels.
        Dense(784, 256, initialisation="he_normal"),
        ReLU(),

        Dense(256, 128, initialisation="he_normal"),
        ReLU(),

        # Output one probability for each digit class 0–9.
        Dense(128, 10, initialisation="xavier_normal"),
        Softmax(),
    ],
    batch_size=64,
    Loss_Algorithm=CrossEntropyLoss(),
    Optimiser=Adam(learning_rate=0.001, beta1=0.9, beta2=0.999),
    Metrics=(CategoricalAccuracy(),),
)


model.train( # Contains the settings used during the training process.
    x_train,
    y_train,
    epochs=15,
    normalise=True,
    validation_data=(x_test, y_test),
    CallBacks=[
        ProgressBar(mode="epoch", bar_length=50),
        EarlyStopping(monitor="val_loss", patience=3),
        CSVLogger("dense_results.csv"),
    ],
)

# Evaluate the final model on unseen test images.
model.test(x_test, y_test)

# Save the trained parameters for later use.
model.save("dense_mnist_model.npz")