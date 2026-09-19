import numpy as np
from .base import Layer

class Activation(Layer):
    def __init__(self):
        super().__init__()

    def forward(self):
        raise NotImplementedError
    
    def backward(self):
        raise NotImplementedError

class Linear(Activation):
    def forward(self, activations, training):
        return activations
    
    def backward(self, dC_da):
        return dC_da

class ReLU(Activation):
    def forward(self, activations, training):
        self.activations = activations
        self.output = np.maximum(0, activations)
        return self.output

    def backward(self, dC_da):
        return (self.activations > 0).astype(int) * dC_da


class Sigmoid(Activation):
    def forward(self, activations, training):
        self.output = (1 / (1 + np.exp(-activations)))
        return self.output

    def backward(self, dC_da):
        return (self.output * (1 - self.output)) * dC_da


class Tanh(Activation):
    def forward(self, activations, training):
        self.output = np.tanh(activations)
        return self.output

    def backward(self, dC_da):
        return (1 - self.output ** 2) * dC_da


class Softmax(Activation):
    def forward(self, activations, training):
        shifted_output = activations - np.max(activations, axis=1, keepdims=True)

        exp = np.exp(shifted_output)
        sums = np.sum(exp, axis=1, keepdims=True)

        self.output = exp / sums

        return self.output

    def backward(self, dC_da): 
        dC_dz = np.zeros_like(self.output)

        for i in range(self.output.shape[0]):
            a = self.output[i]
            jacobean = np.diag(a) - np.outer(a, a)
            dC_dz[i] = jacobean @ dC_da[i]

        return dC_dz

class LeakyReLU(Activation):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, activations, training):
        self.input = activations
        return np.where(activations > 0, activations, self.negative_slope * activations)

    def backward(self, dC_da):
        return dC_da * np.where(self.input > 0, 1, self.negative_slope)

class ELU(Activation):
    def __init__(self, negative_slope=1):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, activations, training):
        self.input = activations
        return np.where(activations > 0, activations, self.negative_slope * (np.exp(activations) - 1))

    def backward(self, dC_da):
        return np.where(self.input > 0, 1, self.negative_slope * np.exp(self.input)) * dC_da

class SELU(Activation):
    def __init__(self):
        super().__init__()
        self.lambda_ = 1.0507009873554805
        self.alpha = 1.6732632423543772

    def forward(self, activations, training):
        self.input = activations
        return np.where(activations > 0, self.lambda_ * activations, self.lambda_ * self.alpha * (np.exp(activations) - 1))

    def backward(self, dC_da):
        return np.where(self.input > 0, self.lambda_, self.lambda_ * self.alpha * np.exp(self.input)) * dC_da

class Softplus(Activation):
    def forward(self, activations, training):
        self.input = activations
        return np.log1p(np.exp(activations))

    def backward(self, dC_da):
        return (1 / (1 + np.exp(-self.input))) * dC_da

class Softsign(Activation):
    def forward(self, activations, training):
        self.input = activations
        return activations / (1 + np.abs(activations))

    def backward(self, dC_da):
        return (1 / (1 + np.abs(self.input)) ** 2) * dC_da

class Swish(Activation):
    def forward(self, activations, training):
        self.input = activations
        self.sigmoid = 1 / (1 + np.exp(-activations))
        return activations * self.sigmoid

    def backward(self, dC_da):
        derivative = self.sigmoid + self.input * self.sigmoid * (1 - self.sigmoid)
        return derivative * dC_da

class GELU(Activation): 
    def forward(self, activations, training):
        self.input = activations
        return 0.5 * activations * (1 + np.tanh(np.sqrt(2 / np.pi) * (activations + 0.044715 * activations ** 3)))

    def backward(self, dC_da):
        x = self.input

        u = np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)
        tanh_u = np.tanh(u)

        derivative = 0.5 * (1 + tanh_u) + 0.5 * x * (1 - tanh_u ** 2) * (np.sqrt(2 / np.pi) * (1 + 3 * (0.044715) * x ** 2))
        return derivative * dC_da

class Mish(Activation):
    def forward(self, activations, training):
        self.input = activations
        self.softplus = np.log1p(np.exp(activations))
        output = activations * np.tanh(self.softplus)

        return output

    def backward(self, dC_da):
        x = self.input
        sigmoid = 1 / (1 + np.exp(-x))
        tanh_sp = np.tanh(self.softplus)

        derivative = tanh_sp + x * (1 - tanh_sp ** 2) * sigmoid

        return derivative * dC_da