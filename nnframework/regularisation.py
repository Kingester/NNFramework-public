import numpy as np

class Regulariser():
    def calculate(self):
        raise NotImplementedError("Not Implemented yet!")
    
    def gradient(self):
        raise NotImplementedError("Not Implemented yet!")

class L1Reg(Regulariser):
    def __init__(self, lambda_):
        self.lambda_ = lambda_

    def calculate(self, weights):
        penalty = np.sum(np.abs(weights))
        return self.lambda_ * penalty

    def gradient(self, weights):
        L1grad = np.sign(weights)

        return self.lambda_ * L1grad


class L2Reg(Regulariser):
    def __init__(self, lambda_):
        self.lambda_ = lambda_

    def calculate(self, weights):
        penalty = np.sum(np.square(weights))
        return self.lambda_ * penalty

    def gradient(self, weights):
        L2grad = 2 * weights

        return self.lambda_ * L2grad

class ElasticNet(Regulariser):
    def __init__(self, L1Lambda, L2Lambda):
        self.L1Lambda = L1Lambda
        self.L2Lambda = L2Lambda
    
    def calculate(self, weights):
        L1Penalty = np.sum(np.abs(weights))
        L2Penalty = np.sum(np.square(weights))

        return self.L1Lambda * L1Penalty + self.L2Lambda * L2Penalty
    
    def gradient(self, weights):
        L1grad = np.sign(weights)
        L2grad = 2 * weights

        return self.L1Lambda * L1grad + self.L2Lambda * L2grad