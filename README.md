# Neural Network Framework


##### An educational neural network framework built from scratch with Python and NumPy.


nn-framework implements the main components of many modern neural network architectures without using any pre-existing machine learning libraries for the model itself. Its purpose is to make the mathematics and software behind neural networks more visible, instead of ignoring the underlying mathematics which are fundamental to understand.


> This is an educational project, not a replacement for PyTorch or TensorFlow!


## Highlights


 - Fully NumPy based neural network implementation.
 - Support for Feedforward networks and convolutional neural networks.
 - Manual forward propagation and backpropagation.
 - Mini batch training.
 - Model saving and loading.
 - MNIST Dense and CNN example models available.


## Components


| Type | Implemented |
|---|---|
| **Layers** | Dense, Conv2D, MaxPool2D, Flatten, BatchNorm1D, BatchNorm2D |
| **Activations** | Linear, Sigmoid, Tanh, ReLU, Leaky ReLU, GELU, Softplus, Softmax, ELU, SELU, Softsign, Swish, Mish |
| **Losses** | MSE, MAE, Cross Entropy, Binary Cross Entropy, Huber, focal losses, hinge losses, KL divergence, Poisson and more |
| **Optimisers** | SGD, Momentum, Nesterov, AdaGrad, RMSProp, Adam, AdamW, Nadam, Adamax, AMSGrad, Lion |
| **Metrics** | Accuracy, Precision, Recall, F1, Top-k Accuracy, RMSE, MAPE, R², Confusion Matrix |
| **Callbacks** | Early Stopping, model checkpoints, CSV logging, learning-rate scheduling, progress bars, NaN detection |
| **Regularisation** | L1, L2 and Elastic Net |


## Installation


In order to install the framework, follow these steps:


 1. Begins by opening **Command Prompt** or the terminal within VS Code.
 2. Clone the repository by entering in the terminal:


```bash
git clone https://github.com/Kingester/NNFramework.git
```
 3. Enter the folder by:


```bash
cd NNFramework
```


 4. Install the required Python packages:


```bash
pip install -r requirements.txt
```


To train the included MNIST model as a test, run:


```bash
python train_mnist.py
```


## Examples


Run the included MNIST training example:


```bash
python train_mnist.py
```


The framework includes an example of a Dense and a CNN architecture.


## Limitations


 - CPU only NumPy implementation.
 - Backpropagation is manually implemented and not vectorised completely.
 - Designed for learning, rather than production deployment.


## Future Work


 - Optional CuPy GPU backend.
 - Additional datasets, including CIFAR-10 and CIFAR-100.
 - More model architectures.




## Lisence


This project is licensed under the MIT License.











