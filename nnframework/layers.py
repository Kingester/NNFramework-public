import numpy as np
from .base import Layer
from .initialisers import Random_Uniform, Random_Normal, He_Uniform, He_Normal, Xavier_Uniform, Xavier_Normal


class Conv2D(Layer):
    def __init__(self, input_channels, num_filters, kernel_size, stride, padding, initialisation="He_Normal"):
        self.trainable = True
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.INITIALISERS = {
            "he_normal": He_Normal,
            "xavier_normal": Xavier_Normal,
            "random_normal": Random_Normal,
            "he_uniform": He_Uniform,
            "xavier_uniform": Xavier_Uniform,
            "random_uniform": Random_Uniform
        }

        if initialisation not in self.INITIALISERS:
            raise ValueError("Unknown initialiser!")

        self.filters = self.INITIALISERS[initialisation].initialisation(
            (num_filters, input_channels, kernel_size, kernel_size)
        )
        
        self.biases = np.zeros(num_filters)

    
    def forward(self, input, training):
        self.input = input

        if len(self.input.shape) != 4:
            self.input = self.input.reshape(self.input.shape[0], 1, self.input.shape[1], self.input.shape[2])

        if self.input.shape[1] != self.input_channels:
            raise ValueError(
                f"Conv2D expected {self.input_channels} input channels, "
                f"but received {self.input.shape[1]}."
        )

        self.padded_input = np.pad(
            self.input,
            (
                (0, 0),                          
                (0, 0),                          
                (self.padding, self.padding),    
                (self.padding, self.padding)     
            )
        )

        self.output_height = int(np.floor((self.input.shape[2] + 2 * self.padding - self.kernel_size) / self.stride + 1))
        self.output_width = int(np.floor((self.input.shape[3] + 2 * self.padding - self.kernel_size) / self.stride + 1))

        patches = np.lib.stride_tricks.sliding_window_view(
            self.padded_input,
            (self.kernel_size, self.kernel_size),
            (2, 3)
        )

        patches = patches[:, :, ::self.stride, ::self.stride, :, :]
        patches = patches.transpose(0, 2, 3, 1, 4, 5)

        patches = patches.reshape(patches.shape[0] * patches.shape[1] * patches.shape[2], -1)
        self.patches = patches

        self.filters_matrix = self.filters.reshape(self.num_filters, -1)

        result = patches @ self.filters_matrix.T
        result = result + self.biases

        result = result.reshape(self.input.shape[0], self.output_height, self.output_width, self.num_filters)
        result = np.transpose(result, (0, 3, 1 ,2))

        return result



    def backward(self, dC_dz):
        dZ = np.transpose(dC_dz, (0, 2, 3, 1))
        dZ = dZ.reshape(-1, self.num_filters)

        self.filter_gradients = dZ.T @ self.patches
        self.filter_gradients = self.filter_gradients.reshape(self.num_filters, self.input_channels, self.kernel_size, self.kernel_size)

        self.bias_gradients = np.sum(dZ, axis=0)

        gradient = dZ @ self.filters_matrix
        gradient = gradient.reshape((self.input.shape[0], self.output_height, self.output_width, self.input_channels, self.kernel_size, self.kernel_size))

        input_gradients = np.zeros_like(self.padded_input)

        for batch in range(dC_dz.shape[0]):
            for row in range(dC_dz.shape[2]):
                input_row = row * self.stride
                for col in range(dC_dz.shape[3]):
                    input_col = col * self.stride

                    input_gradients[batch, :, input_row: input_row + self.kernel_size, input_col: input_col + self.kernel_size] += gradient[batch, row, col]
        
        if self.padding == 0:
            return input_gradients

        return input_gradients[
            :,
            :,
            self.padding: -self.padding,
            self.padding: -self.padding
        ]


class MaxPool2D(Layer):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
    
    def forward(self, input, training):
        self.input = input
        self.batch_size = self.input.shape[0]
        self.max_positions = {}

        patches = np.lib.stride_tricks.sliding_window_view(
            self.input,
            (self.kernel_size, self.kernel_size),
            (2, 3)
        )

        self.output_height = int(np.floor((self.input.shape[2] - self.kernel_size) / self.stride + 1))
        self.output_width = int(np.floor((self.input.shape[3] - self.kernel_size) / self.stride + 1))

        patches = patches[:, :, ::self.stride, ::self.stride, :, :]
        patches = patches.reshape(*patches.shape[:4], -1)

        max_values = np.max(patches, axis=-1)
        self.max_index = np.argmax(patches, axis=-1)

        return max_values



    def backward(self, dC_da):
        d_input = np.zeros_like(self.input)

        max_rows = self.max_index // self.kernel_size
        max_cols = self.max_index % self.kernel_size

        batch_index = np.arange(self.input.shape[0])[:, None, None, None]
        channel_index = np.arange(self.input.shape[1])[None, :, None, None]
        row_index = np.arange(self.output_height)[None, None, :, None]
        col_index = np.arange(self.output_width)[None, None, None, :]

        input_rows = row_index * self.stride + max_rows
        input_cols = col_index * self.stride + max_cols

        np.add.at(
            d_input,
            (
                batch_index,
                channel_index,
                input_rows,
                input_cols
            ),
            dC_da
        )

        return d_input 



                       

class Flatten(Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self, input, training):
        self.input_shape = (input.shape[0], input.shape[1], input.shape[2], input.shape[3])
        self.input = input.reshape(input.shape[0], input.shape[1] * input.shape[2] * input.shape[3])

        return self.input

    def backward(self, gradients):
        array = gradients.reshape(self.input_shape)
        return array


class Dense(Layer):
    def __init__(self, input_size, output_size, initialisation="random"):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.INITIALISERS = {
            "he_normal": He_Normal,
            "xavier_normal": Xavier_Normal,
            "random_normal": Random_Normal,
            "he_uniform": He_Uniform,
            "xavier_uniform": Xavier_Uniform,
            "random_uniform": Random_Uniform
        }

        if initialisation not in self.INITIALISERS:
            raise ValueError("Unknown initialiser!")

        self.weights = self.INITIALISERS[initialisation].initialisation(
            (input_size, output_size)
            )
        
        self.biases = np.zeros((1, output_size))
        self.trainable = True


    def forward(self, layer, training):
        self.input = layer
        self.z = layer @ self.weights + self.biases
        return self.z

    def backward(self, dC_dz):
        self.weight_gradients = self.input.T @ dC_dz
        self.bias_gradients = np.sum(dC_dz, axis=0, keepdims=True)

        return dC_dz @ self.weights.T


class BatchNorm1D(Layer):
    def __init__(self, output_features, momentum=0.9):
        self.output_features = output_features
        self.momentum = momentum
        self.epsilon = 0.00000001
        self.gamma = np.ones(output_features)
        self.beta = np.zeros(output_features)
        self.running_mean = np.zeros(output_features)
        self.running_variance = np.ones(output_features)
        self.trainable = True
    
    def forward(self, activations, training):
        self.batch_size = activations.shape[0]

        if training:
            mean = np.mean(activations, axis=0)
            self.variance = np.var(activations, axis=0)

            self.x_hat = (activations - mean) / np.sqrt(self.variance + self.epsilon)
            y = self.gamma * self.x_hat + self.beta

            self.running_mean = self.momentum * (self.running_mean) + (1 - self.momentum) * mean
            self.running_variance = self.momentum * (self.running_variance) + (1 - self.momentum) * self.variance

            return y

        else:

            x_hat = (activations - self.running_mean) / np.sqrt(self.running_variance + self.epsilon)
            y = self.gamma * x_hat + self.beta

            return y

    def backward(self, dY):
        self.d_beta = np.sum(dY, axis=0)
        self.d_gamma = np.sum(dY * self.x_hat, axis=0)

        dX = (self.gamma / (self.batch_size * np.sqrt(self.variance + self.epsilon))) * (self.batch_size * dY - self.d_beta - self.x_hat * self.d_gamma)

        return dX


class BatchNorm2D(Layer):
    def __init__(self, output_features, momentum=0.9):
        self.output_features = output_features
        self.momentum = momentum
        self.epsilon = 0.0000001
        self.gamma = np.ones(output_features)
        self.beta = np.zeros(output_features)
        self.running_mean = np.zeros(output_features)
        self.running_variance = np.ones(output_features)
        self.trainable = True
    
    def forward(self, activations, training):
        self.batch_size = activations.shape[0]
        self.height = activations.shape[2]
        self.width = activations.shape[3]

        if training:
            mean = np.mean(activations, axis=(0, 2, 3))
            self.variance = np.var(activations, axis=(0, 2, 3))

            mean_reshaped = mean.reshape(1, -1, 1, 1)
            variance_reshaped = self.variance.reshape(1, -1, 1, 1)
            gamma_reshaped = self.gamma.reshape(1, -1, 1, 1)
            beta_reshaped = self.beta.reshape(1, -1, 1, 1)

            self.x_hat = (activations - mean_reshaped) / np.sqrt(variance_reshaped + self.epsilon)
            y = gamma_reshaped * self.x_hat + beta_reshaped

            self.running_mean = self.momentum * (self.running_mean) + (1 - self.momentum) * mean
            self.running_variance = self.momentum * (self.running_variance) + (1 - self.momentum) * self.variance

            return y

        else:
            running_mean = self.running_mean.reshape(1, -1, 1, 1)
            running_variance = self.running_variance.reshape(1, -1, 1, 1)
            gamma = self.gamma.reshape(1, -1, 1, 1)
            beta = self.beta.reshape(1, -1, 1, 1)

            x_hat = (activations - running_mean) / np.sqrt(running_variance + self.epsilon)
            y = gamma * x_hat + beta

            return y

    def backward(self, dY):
        self.d_beta = np.sum(dY, axis= (0, 2, 3))
        self.d_gamma = np.sum(dY * self.x_hat, axis=(0, 2, 3))

        N = self.batch_size * self.height * self.width

        gamma = self.gamma.reshape(1, -1, 1, 1)
        variance = self.variance.reshape(1, -1, 1, 1)
        d_beta = self.d_beta.reshape(1, -1, 1, 1)
        d_gamma = self.d_gamma.reshape(1, -1, 1, 1)

        dX = (gamma / (N * np.sqrt(variance + self.epsilon))) * (N * dY - d_beta - self.x_hat * d_gamma)

        return dX