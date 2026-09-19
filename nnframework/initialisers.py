import numpy as np

def calculate_fans(shape):
    if len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    
    elif len(shape) == 4:
        num_filters, input_channels, kernel_height, kernel_width = shape

        receptive_field_size = kernel_height * kernel_width

        fan_in = receptive_field_size * input_channels
        fan_out = receptive_field_size * num_filters
    
    else:
        raise ValueError(
            "Initlialiser only supports shape of 2 or 4!"
        )
    
    return fan_in, fan_out

class He_Normal():
    @staticmethod
    def initialisation(shape):
        fan_in, _ = calculate_fans(shape)

        return np.random.randn(*shape) * np.sqrt(2 / fan_in)

class He_Uniform(): 
    @staticmethod
    def initialisation(shape):
        fan_in, _ = calculate_fans(shape)

        limit = np.sqrt(6 / fan_in)

        return np.random.uniform(-limit, limit, shape)

class Xavier_Normal():
    @staticmethod
    def initialisation(shape):
        fan_in, fan_out = calculate_fans(shape)

        return np.random.randn(*shape) * np.sqrt(2 / (fan_in + fan_out))

class Xavier_Uniform(): 
    @staticmethod
    def initialisation(shape):
        fan_in, fan_out = calculate_fans(shape)

        limit = np.sqrt(6 / (fan_in + fan_out))

        return np.random.uniform(-limit, limit, shape)

class Random_Normal():
    @staticmethod
    def initialisation(shape):
        return np.random.randn(*shape) * 0.01

class Random_Uniform():
    @staticmethod
    def initialisation(shape):
        return np.random.uniform(-0.01, 0.01, shape)