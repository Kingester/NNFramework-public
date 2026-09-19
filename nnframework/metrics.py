import numpy as np

class Metrics():
    is_matrix = False
    def calculate():
        raise NotImplementedError

class CategoricalAccuracy(Metrics):   
    name = "Accuracy"
    display_format = ".2%"

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, prediction, target):
        prediction_class = np.argmax(prediction, axis=1)
        target_class = np.argmax(target, axis=1)

        self.correct += np.sum(prediction_class == target_class)
        self.total += prediction.shape[0]
    
    def calculate(self):
        return self.correct / self.total
    

class BinaryAccuracy(Metrics):
    name = "Accuracy"
    display_format = ".2%"

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, prediction, target):
        self.correct = np.sum(prediction == target)
        self.total += prediction.shape[0]
    
    def calculate(self):
        return self.correct / self.total

class MeanSE(Metrics):
    name = "MSE"
    display_format = ".6f"

    def reset(self):
        self.total_error = 0
        self.total = 0

    def update(self, prediction, target):
        self.total_error += np.mean((prediction - target) ** 2)
        self.total += prediction.shape[0]
    
    def calculate(self):
        return self.total_error / self.total
    

class MeanAE(Metrics):
    name = "MAE"
    display_format = ".4f"

    def reset(self):
        self.total_error = 0
        self.total = 0

    def update(self, prediction, target):
        self.total_error += np.mean(np.abs(prediction - target))
        self.total += prediction.shape[0]
    
    def calculate(self):
        return self.total_error / self.total


class Precision(Metrics):
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.name = "Precision"
        self.display_format = ".2%"
        self.reset()
    
    def reset(self):
        self.true_prediction = 0
        self.false_prediction = 0

    def update(self, prediction, target):
        prediction_class = (prediction >= self.threshold)
        target_class = (target == 1)

        self.true_prediction += np.sum((prediction_class == 1) & (target_class == 1))
        self.false_prediction += np.sum((prediction_class == 1) & (target_class == 0))
    
    def calculate(self):
        if self.true_prediction + self.false_prediction == 0:
            return 0
        
        return self.true_prediction / (self.true_prediction + self.false_prediction)

class Recall(Metrics):
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        name = "Recall"
        display_format = ".2%"
        self.reset()

    def reset(self):
        self.true_prediction = 0
        self.false_negative = 0

    def update(self, prediction, target):
        prediction_class = (prediction >= self.threshold)
        target_class = (target == 1)

        self.true_prediction += np.sum((prediction_class == 1) & (target_class == 1))
        self.false_negative += np.sum((prediction_class == 0) & (target_class == 1))

    def calculate(self):
        if self.true_prediction + self.false_negative == 0:
            return 0
        
        return self.true_prediction / (self.true_prediction + self.false_negative)

class F1Score(Metrics):
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        name = "F1Score"
        display_format = ".4f"
    
    def reset(self):
        self.true_prediction = 0
        self.false_prediction = 0
        self.false_negative = 0

    def update(self, prediction, target):
        prediction_class = (prediction >= self.threshold)
        target_class = (target == 1)

        self.true_prediction += np.sum((prediction_class == 1) & (target_class == 1))
        self.false_prediction += np.sum((prediction_class == 1) * (target_class == 0))
        self.false_negative += np.sum((prediction_class == 0) & (target_class == 1))

    def calculate(self):
        if self.true_prediction + self.false_negative == 0:
            recall = 0
        else:
            recall = self.true_prediction / (self.true_prediction + self.false_negative)
        
        if self.true_prediction + self.false_prediction == 0:
            precision = 0
        else:
            precision = self.true_prediction / (self.true_prediction + self.false_prediction)
        
        if recall + precision == 0:
            return 0

        return (2 * precision * recall) / (precision + recall)

class ConfusionMatrix(Metrics):
    def __init__(self, size=10):
        self.size = size
        self.name = "Confusion Matrix"
        self.is_matrix = True

    def reset(self):
        self.confusion_matrix = np.zeros((self.size, self.size), dtype=int)

    def update(self, prediction, target):
        self.prediction_class = np.argmax(prediction, axis=1)
        self.target_class = np.argmax(target, axis=1)

        for i in range(prediction.shape[0]):
            self.confusion_matrix[self.target_class[i]][self.prediction_class[i]] += 1
    
    def calculate(self):
        return self.confusion_matrix


class TopKCatergoicalAccuracy(Metrics):
    def __init__(self, Topk=3):
        self.Topk = Topk
        self.display_format = ".2%"
        self.name = f"Top-{Topk} Accuracy"

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, prediction, target):
        top = np.argsort(prediction, axis=1)[:, -self.Topk:][:, ::-1]
        target_class = np.argmax(target, axis=1)

        for i in range(len(top)):   
            if target_class[i] in top[i]:
                self.correct += 1
        
        self.total += len(top)

    def calculate(self):
        return self.correct / self.total

class RootMeanSquaredError(Metrics):
    display_format = ".4f"
    name = "RMSE"

    def reset(self):
        self.total_error = 0
        self.total = 0

    def update(self, prediction, target):
        self.total_error += np.sqrt(np.mean((prediction - target) ** 2))
        self.total += prediction.shape[0]
    
    def calculate(self):
        return self.total_error / self.total

class MeanAbsolutePercentageError(Metrics):
    display_format = ".2f"
    name = "MAPE"

    def reset(self):
        self.total_error = 0
        self.total = 0

    def update(self, prediction, target):
        epsilon = 0.0000001

        self.total_error += np.sum(np.abs(target - prediction) / (np.abs(target) + epsilon))
        self.total += prediction.shape[0]
    
    def calculate(self):
        return (self.total_error / self.total) * 100

class R2Score(Metrics):
    display_format = ".6f"
    name = "R2 Score"

    def reset(self):
        self.predictions = []
        self.targets = []

    def update(self, prediction, target):
        self.predictions.append(prediction)
        self.targets.append(target)

    def calculate(self):
        self.predictions = self.predictions.flatten()
        self.targets = self.targets.flatten()

        epsilon = 0.00000001

        SS_res = np.sum((self.targets - self.predictions) ** 2)

        mean_target = np.mean(self.targets)
        SS_tot = np.sum((self.targets - mean_target) ** 2)

        return 1 - (SS_res / (SS_tot + epsilon))