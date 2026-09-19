import numpy as np
from .layers import Conv2D, Dense, BatchNorm1D, BatchNorm2D


class Optimiser:
    def step(self):
        raise NotImplementedError("Step not implemented yet!")

class SGD(Optimiser):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def step(self, layers):
        for layer in layers:
            if layer.trainable:

                if isinstance(layer, Conv2D):
                    layer.filters = layer.filters - layer.filter_gradients * self.learning_rate
                    layer.biases = layer.biases - layer.bias_gradients * self.learning_rate
                elif isinstance(layer, Dense):
                    layer.weights = layer.weights - layer.weight_gradients * self.learning_rate
                    layer.biases = layer.biases - layer.bias_gradients * self.learning_rate     
                elif isinstance(layer, BatchNorm1D) or isinstance(layer, BatchNorm2D):
                    layer.gamma = layer.gamma - layer.d_gamma * self.learning_rate
                    layer.beta = layer.beta - layer.d_beta * self.learning_rate              


class Adam(Optimiser):
    def __init__(self, learning_rate, beta1, beta2):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = 0.00000001
        self.t = 0
        self.states = {}

    def step(self, layers):
        self.t += 1

        for layer in layers:
            if layer.trainable:
                if isinstance(layer, Dense) or isinstance(layer, Conv2D):
                    if isinstance(layer, Conv2D):
                        weights = layer.filters
                        weight_gradients = layer.filter_gradients
                        bias_gradients = layer.bias_gradients
                        biases = layer.biases
                    else:
                        weights = layer.weights
                        weight_gradients = layer.weight_gradients
                        bias_gradients = layer.bias_gradients
                        biases = layer.biases
                
                if isinstance(layer, BatchNorm1D) or isinstance(layer, BatchNorm2D):
                    weights = layer.gamma
                    weight_gradients = layer.d_gamma
                    biases = layer.beta
                    bias_gradients = layer.d_beta

                if id(layer) not in self.states:
                    self.states[id(layer)] = {
                        "m_weights": np.zeros_like(weights),
                        "v_weights": np.zeros_like(weights),
                        "m_bias": np.zeros_like(biases),
                        "v_bias": np.zeros_like(biases)
                    }

                state = self.states[id(layer)]
                
                m_weights = state["m_weights"]
                m_bias = state["m_bias"]
                v_weights = state["v_weights"]
                v_bias = state["v_bias"]

                m_weights = self.beta1 * m_weights + (1 - self.beta1) * weight_gradients
                v_weights = self.beta2 * v_weights + (1 - self.beta2) * (weight_gradients ** 2)

                m_bias = self.beta1 * m_bias + (1 - self.beta1) * bias_gradients
                v_bias = self.beta2 * v_bias + (1 - self.beta2) * (bias_gradients ** 2)

                m_hat_w = m_weights / (1 - self.beta1 ** self.t)
                v_hat_w = v_weights / (1 - self.beta2 ** self.t)

                m_hat_b = m_bias / (1 - self.beta1 ** self.t)
                v_hat_b = v_bias / (1 - self.beta2 ** self.t)

                if isinstance(layer, Conv2D):
                    layer.filters = layer.filters - (m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)) * self.learning_rate
                    layer.biases = layer.biases - (m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)) * self.learning_rate
                elif isinstance(layer, Dense):
                    layer.weights = layer.weights - (m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)) * self.learning_rate
                    layer.biases = layer.biases - (m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)) * self.learning_rate
                elif isinstance(layer, BatchNorm1D) or isinstance(layer, BatchNorm2D):
                    layer.gamma = layer.gamma - (m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)) * self.learning_rate
                    layer.beta = layer.beta - (m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)) * self.learning_rate

                state["m_weights"] = m_weights
                state["m_bias"] = m_bias
                state["v_weights"] = v_weights
                state["v_bias"] = v_bias



class Momentum(Optimiser):  
    def __init__(self, learning_rate, beta1):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.states = {}
    
    def step(self, layers):
        for layer in layers:
            if layer.trainable:

                if isinstance(layer, Dense) or isinstance(layer, Conv2D):
                    if isinstance(layer, Conv2D):
                        weights = layer.filters
                        weight_gradients = layer.filter_gradients
                        bias_gradients = layer.bias_gradients
                        biases = layer.biases
                    else:
                        weights = layer.weights
                        weight_gradients = layer.weight_gradients
                        bias_gradients = layer.bias_gradients
                        biases = layer.biases

                if isinstance(layer, BatchNorm1D) or isinstance(layer, BatchNorm2D):
                    weights = layer.gamma
                    weight_gradients = layer.d_gamma
                    biases = layer.beta
                    bias_gradients = layer.d_beta                

                if id(layer) not in self.states:
                    self.states[id(layer)] = {
                        "v_weights": np.zeros_like(weights),
                        "v_bias": np.zeros_like(biases)
                    }
                
                state = self.states[id(layer)]

                v_weights = state["v_weights"]
                v_bias = state["v_bias"]

                v_weights = self.beta1 * v_weights  + (1 - self.beta1) * weight_gradients
                v_bias = self.beta1 * v_bias + (1 - self.beta1) * bias_gradients

                if isinstance(layer, Conv2D):
                    layer.filters = layer.filters - self.learning_rate * v_weights
                    layer.biases = layer.biases - self.learning_rate * v_bias
                elif isinstance(layer, Dense):
                    layer.weights = layer.weights - self.learning_rate * v_weights
                    layer.biases = layer.biases - self.learning_rate * v_bias
                elif isinstance(layer, BatchNorm1D) or isinstance(layer, BatchNorm2D):
                    layer.gamma = layer.gamma - self.learning_rate * v_weights
                    layer.beta = layer.beta - self.learning_rate * v_bias


                state["v_weights"] = v_weights
                state["v_bias"] = v_bias


class Nesterov(Optimiser): 
    def __init__(self, learning_rate, beta1):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.states = {}
    
    def step(self, layers):
        for layer in layers:
            if layer.trainable:

                if isinstance(layer, Dense) or isinstance(layer, Conv2D):
                    if isinstance(layer, Conv2D):
                        weights = layer.filters
                        weight_gradients = layer.filter_gradients
                        bias_gradients = layer.bias_gradients
                        biases = layer.biases
                    else:
                        weights = layer.weights
                        weight_gradients = layer.weight_gradients
                        bias_gradients = layer.bias_gradients
                        biases = layer.biases
                
                if isinstance(layer, BatchNorm1D) or isinstance(layer, BatchNorm2D):
                    weights = layer.gamma
                    weight_gradients = layer.d_gamma
                    biases = layer.beta
                    bias_gradients = layer.d_beta                

                if id(layer) not in self.states:
                    self.states[id(layer)] = {
                        "v_weights": np.zeros_like(weights),
                        "v_bias": np.zeros_like(biases)                        
                    }
                
                state = self.states[id(layer)]

                previous_v_weights = state["v_weights"].copy()
                previous_v_bias = state["v_bias"].copy()

                v_weights = ( self.beta1 * previous_v_weights - self.learning_rate * weight_gradients)

                v_bias = (self.beta1 * previous_v_bias - self.learning_rate * bias_gradients)

                weight_update = ( -self.beta1 * previous_v_weights + (1 + self.beta1) * v_weights)

                bias_update = ( -self.beta1 * previous_v_bias + (1 + self.beta1) * v_bias)

                if isinstance(layer, Conv2D):
                    layer.filters = layer.filters + weight_update
                    layer.biases = layer.biases + bias_update

                elif isinstance(layer, Dense):
                    layer.weights = layer.weights + weight_update
                    layer.biases = layer.biases + bias_update

                elif isinstance(layer, BatchNorm1D) or isinstance(layer, BatchNorm2D):
                    layer.gamma = layer.gamma + weight_update
                    layer.beta = layer.beta + bias_update

                state["v_weights"] = v_weights
                state["v_bias"] = v_bias              

class AdaGrad(Optimiser):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.states = {}
        self.epsilon = 0.00000001
    
    def step(self, layers):
        for layer in layers:
            if layer.trainable:
                if isinstance(layer, Dense) or isinstance(layer, Conv2D):
                    if isinstance(layer, Conv2D):
                        weights = layer.filters
                        weight_gradients = layer.filter_gradients
                        bias_gradients = layer.bias_gradients
                        biases = layer.biases
                    else:
                        weights = layer.weights
                        weight_gradients = layer.weight_gradients
                        bias_gradients = layer.bias_gradients
                        biases = layer.biases

                if isinstance(layer, BatchNorm1D) or isinstance(layer, BatchNorm2D):
                    weights = layer.gamma
                    weight_gradients = layer.d_gamma
                    biases = layer.beta
                    bias_gradients = layer.d_beta

                if id(layer) not in self.states:
                    self.states[id(layer)] = {
                        "weight_cache": np.zeros_like(weights),
                        "bias_cache": np.zeros_like(biases)
                    }
                
                state = self.states[id(layer)]
                weight_cache = state["weight_cache"]
                bias_cache = state["bias_cache"]

                weight_cache += weight_gradients ** 2
                bias_cache += bias_gradients ** 2

                if isinstance(layer, Conv2D):
                    layer.filters = layer.filters - (self.learning_rate * weight_gradients) / np.sqrt(weight_cache + self.epsilon)
                    layer.biases = layer.biases - (self.learning_rate * bias_gradients) / np.sqrt(bias_cache + self.epsilon)
                elif isinstance(layer, Dense):
                    layer.weights = layer.weights - (self.learning_rate * weight_gradients) / np.sqrt(weight_cache + self.epsilon)
                    layer.biases = layer.biases - (self.learning_rate * bias_gradients) / np.sqrt(bias_cache + self.epsilon)
                elif isinstance(layer, BatchNorm1D) or isinstance(layer, BatchNorm2D):
                    layer.gamma = layer.gamma - (self.learning_rate * weight_gradients) / np.sqrt(weight_cache + self.epsilon)
                    layer.beta = layer.beta - (self.learning_rate * bias_gradients) / np.sqrt(bias_cache + self.epsilon)


                state["weight_cache"] = weight_cache
                state["bias_cache"] = bias_cache



class RMSProp(Optimiser):
    def __init__(self, learning_rate, decay):
        self.learning_rate = learning_rate
        self.decay = decay
        self.epsilon = 0.00000001
        self.states = {}
    
    def step(self, layers):
        for layer in layers:
            if layer.trainable:
                if isinstance(layer, Dense) or isinstance(layer, Conv2D):
                    if isinstance(layer, Conv2D):
                        weights = layer.filters
                        weight_gradients = layer.filter_gradients
                        bias_gradients = layer.bias_gradients
                        biases = layer.biases
                    else:
                        weights = layer.weights
                        weight_gradients = layer.weight_gradients
                        bias_gradients = layer.bias_gradients
                        biases = layer.biases

                if isinstance(layer, BatchNorm1D) or isinstance(layer, BatchNorm2D):
                    weights = layer.gamma
                    weight_gradients = layer.d_gamma
                    biases = layer.beta
                    bias_gradients = layer.d_beta

                if id(layer) not in self.states:
                    self.states[id(layer)] = {
                        "weight_cache": np.zeros_like(weights),
                        "bias_cache": np.zeros_like(biases)
                    }
                
                states = self.states[id(layer)]
                weight_cache = states["weight_cache"]
                bias_cache = states["bias_cache"]

                weight_cache = self.decay * weight_cache + (1 - self.decay) * (weight_gradients ** 2)
                bias_cache = self.decay * bias_cache + (1 - self.decay) * (bias_gradients ** 2)

                if isinstance(layer, Conv2D):
                    layer.filters = layer.filters - self.learning_rate * weight_gradients / np.sqrt(weight_cache + self.epsilon)
                    layer.biases = layer.biases - self.learning_rate * bias_gradients / np.sqrt(bias_cache + self.epsilon)
                elif isinstance(layer, Dense):
                    layer.weights = layer.weights - self.learning_rate * weight_gradients / np.sqrt(weight_cache + self.epsilon)
                    layer.biases = layer.biases - self.learning_rate * bias_gradients / np.sqrt(bias_cache + self.epsilon)
                elif isinstance(layer, BatchNorm1D) or isinstance(layer, BatchNorm2D):
                    layer.gamma = layer.gamma - self.learning_rate * weight_gradients / np.sqrt(weight_cache + self.epsilon)
                    layer.beta = layer.beta - self.learning_rate * bias_gradients / np.sqrt(bias_cache + self.epsilon)

                states["weight_cache"] = weight_cache
                states["bias_cache"] = bias_cache


class AdamW(Optimiser):
    def __init__(self, learning_rate, beta1, beta2, decay):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.decay = decay
        self.epsilon = 0.00000001
        self.states = {}
        self.t = 0
    
    def step(self, layers):
        self.t += 1
        for layer in layers:
            if layer.trainable:
                if isinstance(layer, Dense) or isinstance(layer, Conv2D):
                    if isinstance(layer, Conv2D):
                        weights = layer.filters
                        weight_gradients = layer.filter_gradients
                        bias_gradients = layer.bias_gradients
                        biases = layer.biases
                    else:
                        weights = layer.weights
                        weight_gradients = layer.weight_gradients
                        bias_gradients = layer.bias_gradients
                        biases = layer.biases
                    
                if isinstance(layer, BatchNorm1D) or isinstance(layer, BatchNorm2D):
                    weights = layer.gamma
                    weight_gradients = layer.d_gamma
                    biases = layer.beta
                    bias_gradients = layer.d_beta

                if id(layer) not in self.states:
                    self.states[id(layer)] = {
                        "m_weights": np.zeros_like(weights),
                        "v_weights": np.zeros_like(weights),
                        "m_bias": np.zeros_like(biases),
                        "v_bias": np.zeros_like(biases)
                    }
                state = self.states[id(layer)]

                m_weights = state["m_weights"]
                m_bias = state["m_bias"]
                v_weights = state["v_weights"]
                v_bias = state["v_bias"]

                m_weights = self.beta1 * m_weights + (1 - self.beta1) * weight_gradients
                v_weights = self.beta2 * v_weights + (1 - self.beta2) * (weight_gradients * weight_gradients)

                m_bias = self.beta1 * m_bias + (1 - self.beta1) * bias_gradients
                v_bias = self.beta2 * v_bias + (1 - self.beta2) * (bias_gradients * bias_gradients)

                m_hat_w = m_weights / (1 - self.beta1 ** self.t)
                v_hat_w = v_weights / (1 - self.beta2 ** self.t)

                m_hat_b = m_bias / (1 - self.beta1 ** self.t)
                v_hat_b = v_bias / (1 - self.beta2 ** self.t)

                if isinstance(layer, Conv2D):
                    layer.filters = layer.filters - self.learning_rate * ((m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)) + self.decay * layer.filters)
                    layer.biases = layer.biases - self.learning_rate * ((m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)) + self.decay * layer.biases)
                elif isinstance(layer, Dense):
                    layer.weights = layer.weights - self.learning_rate * ((m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)) + self.decay * layer.weights)
                    layer.biases = layer.biases - self.learning_rate * ((m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)) + self.decay * layer.biases)      
                elif isinstance(layer, BatchNorm1D) or isinstance(layer, BatchNorm2D):
                    layer.gamma = layer.gamma - self.learning_rate * ((m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)) + self.decay * layer.gamma)   
                    layer.beta = layer.beta - self.learning_rate * ((m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)) + self.decay * layer.beta)         

                state["m_weights"] = m_weights
                state["m_bias"] = m_bias
                state["v_weights"] = v_weights
                state["v_bias"] = v_bias                



class Nadam(Optimiser):
    def __init__(self, learning_rate, beta1, beta2):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = 0.00000001
        self.states = {}
        self.t = 0
    
    def step(self, layers):
        self.t += 1
        for layer in layers:
            if layer.trainable:
                if isinstance(layer, Dense) or isinstance(layer, Conv2D):
                    if isinstance(layer, Conv2D):
                        weights = layer.filters
                        weight_gradients = layer.filter_gradients
                        bias_gradients = layer.bias_gradients
                        biases = layer.biases
                    else:
                        weights = layer.weights
                        weight_gradients = layer.weight_gradients
                        bias_gradients = layer.bias_gradients
                        biases = layer.biases

                if isinstance(layer, BatchNorm1D) or isinstance(layer, BatchNorm2D):
                    weights = layer.gamma
                    weight_gradients = layer.d_gamma
                    biases = layer.beta
                    bias_gradients = layer.d_beta

                if id(layer) not in self.states:
                    self.states[id(layer)] = {
                        "m_weights": np.zeros_like(weights),
                        "v_weights": np.zeros_like(weights),
                        "m_bias": np.zeros_like(biases),
                        "v_bias": np.zeros_like(biases)
                    }

                state = self.states[id(layer)]

                m_weights = state["m_weights"]
                m_bias = state["m_bias"]
                v_weights = state["v_weights"]
                v_bias = state["v_bias"]                
                
                m_weights = self.beta1 * m_weights + (1 - self.beta1) * weight_gradients
                v_weights = self.beta2 * v_weights + (1 - self.beta2) * weight_gradients ** 2

                m_bias = self.beta1 * m_bias + (1 - self.beta1) * bias_gradients
                v_bias = self.beta2 * v_bias + (1 - self.beta2) * bias_gradients ** 2

                m_hat_w = m_weights / (1 - self.beta1 ** self.t)
                v_hat_w = v_weights / (1 - self.beta2 ** self.t)

                m_hat_b = m_bias / (1 - self.beta1 ** self.t)
                v_hat_b = v_bias / (1 - self.beta2 ** self.t)

                if isinstance(layer, Conv2D):
                    layer.filters = layer.filters - self.learning_rate * ((self.beta1 * m_hat_w + ((1 - self.beta1) / (1 - self.beta1 ** self.t)) * weight_gradients) / (np.sqrt(v_hat_w) + self.epsilon))
                    layer.biases = layer.biases - self.learning_rate * ((self.beta1 * m_hat_b + ((1 - self.beta1) / (1 - self.beta1 ** self.t)) * bias_gradients) / (np.sqrt(v_hat_b) + self.epsilon))    
                elif isinstance(layer, Dense):
                    layer.weights = layer.weights - self.learning_rate * ((self.beta1 * m_hat_w + ((1 - self.beta1) / (1 - self.beta1 ** self.t)) * weight_gradients) / (np.sqrt(v_hat_w) + self.epsilon))
                    layer.biases = layer.biases - self.learning_rate * ((self.beta1 * m_hat_b + ((1 - self.beta1) / (1 - self.beta1 ** self.t)) * bias_gradients) / (np.sqrt(v_hat_b) + self.epsilon))   
                elif isinstance(layer, BatchNorm1D) or isinstance(layer, BatchNorm2D):
                    layer.gamma = layer.gamma - self.learning_rate * ((self.beta1 * m_hat_w + ((1 - self.beta1) / (1 - self.beta1 ** self.t)) * weight_gradients) / (np.sqrt(v_hat_w) + self.epsilon))
                    layer.beta = layer.beta - self.learning_rate * ((self.beta1 * m_hat_b + ((1 - self.beta1) / (1 - self.beta1 ** self.t)) * bias_gradients) / (np.sqrt(v_hat_b) + self.epsilon))  


                state["m_weights"] = m_weights
                state["m_bias"] = m_bias
                state["v_weights"] = v_weights
                state["v_bias"] = v_bias 


class Adamax(Optimiser):
    def __init__(self, learning_rate, beta1, beta2):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = 0.00000001
        self.states = {}
        self.t = 0
    
    def step(self, layers):
        self.t += 1
        for layer in layers:
            if layer.trainable:
                if isinstance(layer, Dense) or isinstance(layer, Conv2D):
                    if isinstance(layer, Conv2D):
                        weights = layer.filters
                        weight_gradients = layer.filter_gradients
                        bias_gradients = layer.bias_gradients
                        biases = layer.biases
                    else:
                        weights = layer.weights
                        weight_gradients = layer.weight_gradients
                        bias_gradients = layer.bias_gradients
                        biases = layer.biases
                    
                if isinstance(layer, BatchNorm1D) or isinstance(layer, BatchNorm2D):
                    weights = layer.gamma
                    weight_gradients = layer.d_gamma
                    biases = layer.beta
                    bias_gradients = layer.d_beta

                if id(layer) not in self.states:
                    self.states[id(layer)] = {
                        "m_weights": np.zeros_like(weights),
                        "u_weights": np.zeros_like(weights), 
                        "m_bias": np.zeros_like(biases),
                        "u_bias": np.zeros_like(biases)
                    }
                
                state = self.states[id(layer)]

                m_weights = state["m_weights"]
                m_bias = state["m_bias"]
                u_weights = state["u_weights"]
                u_bias = state["u_bias"]

                m_weights = self.beta1 * m_weights + (1 - self.beta1) * weight_gradients
                u_weights = np.maximum(self.beta2 * u_weights, np.abs(weight_gradients))

                m_bias = self.beta1 * m_bias + (1 - self.beta1) * bias_gradients
                u_bias = np.maximum(self.beta2 * u_bias, np.abs(bias_gradients))

                m_hat_w = m_weights / (1 - self.beta1 ** self.t)
                m_hat_b = m_bias / (1 - self.beta1 ** self.t)

                if isinstance(layer, Conv2D):
                    layer.filters = layer.filters - self.learning_rate * (m_hat_w / (u_weights + self.epsilon))
                    layer.biases = layer.biases - self.learning_rate * (m_hat_b / (u_bias + self.epsilon))
                elif isinstance(layer, Dense):
                    layer.weights = layer.weights - self.learning_rate * (m_hat_w / (u_weights + self.epsilon))
                    layer.biases = layer.biases - self.learning_rate * (m_hat_b / (u_bias + self.epsilon))
                elif isinstance(layer, BatchNorm1D) or isinstance(layer, BatchNorm2D):
                    layer.gamma = layer.gamma - self.learning_rate * (m_hat_w / (u_weights + self.epsilon))
                    layer.beta = layer.beta - self.learning_rate * (m_hat_b / (u_bias + self.epsilon))

                state["m_weights"] = m_weights
                state["m_bias"] = m_bias
                state["u_weights"] = u_weights
                state["u_bias"] = u_bias 



class AMSGrad(Optimiser):
    def __init__(self, learning_rate, beta1, beta2):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = 0.00000001
        self.states = {}
        self.t = 0

    def step(self, layers):
        self.t += 1

        for layer in layers:
            if layer.trainable:

                if isinstance(layer, Conv2D):
                    weights = layer.filters
                    weight_gradients = layer.filter_gradients
                    biases = layer.biases
                    bias_gradients = layer.bias_gradients

                elif isinstance(layer, Dense):
                    weights = layer.weights
                    weight_gradients = layer.weight_gradients
                    biases = layer.biases
                    bias_gradients = layer.bias_gradients

                elif isinstance(layer, BatchNorm1D) or isinstance(layer, BatchNorm2D):
                    weights = layer.gamma
                    weight_gradients = layer.d_gamma
                    biases = layer.beta
                    bias_gradients = layer.d_beta

                if id(layer) not in self.states:
                    self.states[id(layer)] = {
                        "m_weights": np.zeros_like(weights),
                        "v_weights": np.zeros_like(weights),
                        "v_w_max": np.zeros_like(weights),

                        "m_bias": np.zeros_like(biases),
                        "v_bias": np.zeros_like(biases),
                        "v_b_max": np.zeros_like(biases)
                    }

                state = self.states[id(layer)]

                m_weights = state["m_weights"]
                v_weights = state["v_weights"]
                v_w_max = state["v_w_max"]

                m_bias = state["m_bias"]
                v_bias = state["v_bias"]
                v_b_max = state["v_b_max"]

                m_weights = (
                    self.beta1 * m_weights
                    + (1 - self.beta1) * weight_gradients
                )

                m_bias = (
                    self.beta1 * m_bias
                    + (1 - self.beta1) * bias_gradients
                )

                v_weights = (
                    self.beta2 * v_weights
                    + (1 - self.beta2) * weight_gradients ** 2
                )

                v_bias = (
                    self.beta2 * v_bias
                    + (1 - self.beta2) * bias_gradients ** 2
                )

                v_w_max = np.maximum(v_w_max, v_weights)
                v_b_max = np.maximum(v_b_max, v_bias)

                m_hat_w = m_weights / (1 - self.beta1 ** self.t)
                m_hat_b = m_bias / (1 - self.beta1 ** self.t)

                v_hat_w = v_w_max / (1 - self.beta2 ** self.t)
                v_hat_b = v_b_max / (1 - self.beta2 ** self.t)

                if isinstance(layer, Conv2D):
                    layer.filters = layer.filters - self.learning_rate * (
                        m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
                    )

                    layer.biases = layer.biases - self.learning_rate * (
                        m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)
                    )

                elif isinstance(layer, Dense):
                    layer.weights = layer.weights - self.learning_rate * (
                        m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
                    )

                    layer.biases = layer.biases - self.learning_rate * (
                        m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)
                    )

                elif isinstance(layer, BatchNorm1D) or isinstance(layer, BatchNorm2D):
                    layer.gamma = layer.gamma - self.learning_rate * (
                        m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
                    )

                    layer.beta = layer.beta - self.learning_rate * (
                        m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)
                    )

                state["m_weights"] = m_weights
                state["v_weights"] = v_weights
                state["v_w_max"] = v_w_max

                state["m_bias"] = m_bias
                state["v_bias"] = v_bias
                state["v_b_max"] = v_b_max



class Lion(Optimiser):
    def __init__(self, learning_rate, beta1, beta2):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.states = {}
    
    def step(self, layers):
        for layer in layers:
            if layer.trainable:
                if isinstance(layer, Dense) or isinstance(layer, Conv2D):
                    if isinstance(layer, Conv2D):
                        weights = layer.filters
                        weight_gradients = layer.filter_gradients
                        bias_gradients = layer.bias_gradients
                        biases = layer.biases
                    else:
                        weights = layer.weights
                        weight_gradients = layer.weight_gradients
                        bias_gradients = layer.bias_gradients
                        biases = layer.biases

                if isinstance(layer, BatchNorm1D) or isinstance(layer, BatchNorm2D):
                    weights = layer.gamma
                    weight_gradients = layer.d_gamma
                    biases = layer.beta
                    bias_gradients = layer.d_beta

                if id(layer) not in self.states:
                    self.states[id(layer)] = {
                        "m_weights": np.zeros_like(weights),
                        "m_bias": np.zeros_like(biases)
                    }

                state = self.states[id(layer)]
                m_weights = state["m_weights"]
                m_bias = state["m_bias"]

                c_w = self.beta1 * m_weights + (1 - self.beta1) * weight_gradients
                c_b = self.beta1 * m_bias + (1 - self.beta1) * bias_gradients

                if isinstance(layer, Conv2D):
                    layer.filters = layer.filters - self.learning_rate * np.sign(c_w)
                    layer.biases = layer.biases - self.learning_rate * np.sign(c_b)
                elif isinstance(layer, Dense):
                    layer.weights = layer.weights - self.learning_rate * np.sign(c_w)
                    layer.biases = layer.biases - self.learning_rate * np.sign(c_b)
                elif isinstance(layer, BatchNorm1D) or isinstance(layer, BatchNorm2D):
                    layer.gamma = layer.gamma - self.learning_rate * np.sign(c_w)
                    layer.beta = layer.beta - self.learning_rate * np.sign(c_b)

                m_weights = self.beta2 * m_weights + (1 - self.beta2) * weight_gradients
                m_bias = self.beta2 * m_bias + (1 - self.beta2) * bias_gradients

                state["m_weights"] = m_weights
                state["m_bias"] = m_bias