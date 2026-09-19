import numpy as np
from .base import Layer


class MSE():
    def forward(self, prediction, target):
        self.prediction = prediction
        self.target = target

        self.difference = prediction - target

        return np.sum(np.square(self.difference)) / self.prediction.size

    def backward(self):
        return 2 * (self.difference) / self.prediction.size

class CrossEntropyLoss():
    def forward(self, prediction, target):
        self.prediction = prediction
        self.target = target
        self.epsilon = 0.00000001

        return -np.sum(self.target * np.log(self.prediction + self.epsilon)) / self.prediction.shape[0]

    def backward(self):
        return (self.prediction - self.target) / self.prediction.shape[0]

class BinaryCrossEntropy():
    def forward(self, prediction, target):
        self.prediction = prediction
        self.target = target
        self.epsilon = 0.00000001
        self.loss = -(target * np.log(prediction + self.epsilon) + (1 - target) * np.log(1 - prediction + self.epsilon))

        return np.mean(self.loss)

    def backward(self):
        return (self.prediction - self.target) / self.prediction.size

class MAE():
    def forward(self, prediction, target):
        self.prediction = prediction
        self.target = target

        self.difference = prediction - target

        return np.mean(np.abs(self.difference))

    def backward(self):
        return np.sign(self.difference) / self.prediction.size

class Huber():
    def __init__(self, delta=1):
        self.delta = delta
        
    def forward(self, prediction, target):
        self.prediction = prediction
        self.target = target

        self.e = prediction - target

        return np.mean(np.where(np.abs(self.e) <= self.delta, 0.5 * self.e ** 2, self.delta * (np.abs(self.e) - 0.5 * self.delta)))

    def backward(self):
        return np.where(np.abs(self.e) <= self.delta, self.e, self.delta * np.sign(self.e)) / self.e.size

class KL_Divergence():
    def forward(self, prediction, target):
        self.prediction = prediction
        self.target = target
        self.epsilon = 0.00000001

        return np.sum(self.target * np.log((self.target + self.epsilon) / (self.prediction + self.epsilon))) / self.prediction.shape[0]

    def backward(self):
        return (-self.target / (self.prediction + self.epsilon)) / self.prediction.shape[0]

class SparseCategoricalCrossEntropy():
    def forward(self, prediction, target):
        self.prediction = prediction
        self.target = target
        self.epsilon = 0.00000001

        loss = 0

        for i in range(len(prediction)):
            loss += -np.log(prediction[i][target[i]] + self.epsilon)
        
        return loss / prediction.shape[0]

    def backward(self):
        default = np.zeros_like(self.prediction)
        for i in range(len(self.prediction)):
            default[i][self.target[i]] = 1
        
        return (self.prediction - default) / self.prediction.shape[0]

class Hinge():
    def forward(self, prediction, target):
        self.prediction = prediction
        self.target = target

        self.target = np.where(target > 0, 1, -1)
        return np.mean(np.maximum(0, 1 - self.target * self.prediction))


    def backward(self):
        dC_da = np.where(1 - self.prediction * self.target <= 0, 0, -self.target)

        return dC_da / self.prediction.size

class LogCosh():
    def forward(self, prediction, target):
        self.prediction = prediction
        self.target = target

        self.e = self.prediction - self.target

        return np.mean(np.log(np.cosh(self.e)))

    def backward(self):
        return np.tanh(self.e) / self.prediction.size


class SquaredHinge():
    def forward(self, prediction, target):
        self.prediction = prediction
        self.target = target

        self.target = np.where(target > 0, 1, -1)
        return np.mean((np.maximum(0, 1 - self.prediction * self.target)) ** 2)

    def backward(self):
        return np.where(self.prediction * self.target >= 1, 0, -2 * self.target * (1 - self.target * self.prediction)) / self.prediction.size
    

class CosineSimilarityLoss():
    def forward(self, prediction, target):
        self.prediction = prediction
        self.target = target

        self.prediction_norm = np.sqrt(np.sum(self.prediction ** 2))
        self.target_norm = np.sqrt(np.sum(self.target ** 2))

        self.cosine_similarity = np.sum(self.prediction * self.target) / (self.prediction_norm * self.target_norm)
        return 1 - self.cosine_similarity

    def backward(self):
        return (self.cosine_similarity * self.prediction) / self.prediction_norm ** 2 - self.target / (self.prediction_norm * self.target_norm)

class BinaryFocalLoss():
    def __init__(self, alpha=0.25, gamma=2):
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, prediction, target):
        self.prediction = prediction
        self.target = target
        self.epsilon = 0.00000001

        self.loss = (
            -self.alpha * target * (1 - prediction) ** self.gamma * np.log(prediction + self.epsilon) 
            - (1 - self.alpha) * (1 - target) * prediction ** self.gamma * np.log(1 - prediction + self.epsilon)
            )
        
        return np.mean(self.loss)

    def backward(self):
        p = self.prediction
        a = self.alpha
        g = self.gamma
        y = self.target
        e = self.epsilon

        positive = (
            a * y * (g * (1 - p) ** (g - 1) * np.log(p + e) - (1 - p) ** g / (p + e))
        )

        negative = (
            -(1 - a) * (1 - y) * (g * p ** (g - 1) * np.log(1 - p + e) - p ** g / (1 - p + e))
        )

        return (positive + negative) / self.prediction.shape[0]

class CategoricalFocalLoss():
    def __init__(self, alpha=0.25, gamma=2):
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, prediction, target):
        self.prediction = prediction
        self.target = target
        self.epsilon = 0.00000001

        self.pt = np.sum(self.prediction * self.target, axis=1)
        loss = -self.alpha * (1 - self.pt) ** self.gamma * np.log(self.pt + self.epsilon)

        return np.mean(loss)

    def backward(self):
        a = self.alpha
        g = self.gamma
        y = self.target
        pt = self.pt
        e = self.epsilon

        dL_dp = (
            a * y * (g * (1 - pt) ** (g - 1) * np.log(pt + e)
            - ((1 - pt) ** g) / pt)
            )
        
        return dL_dp / self.prediction.shape[0]
    

class Poisson():
    def forward(self, prediction, target):
        self.prediction = prediction
        self.target = target
        self.epsilon = 0.00000001

        loss = np.sum(self.prediction - self.target * np.log(self.prediction + self.epsilon))
        return loss / self.prediction.shape[0]

    def backward(self):
        return (1 - self.target / (self.prediction + self.epsilon)) / self.prediction.shape[0]