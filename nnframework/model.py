import numpy as np
import time
import math

from .activations import Linear, Sigmoid, Tanh, ReLU, LeakyReLU, GELU, Softplus, Softmax, ELU, SELU, Softsign, Swish, Mish
from .base import Layer
from .losses import MSE, MAE, CrossEntropyLoss, BinaryCrossEntropy, Huber, KL_Divergence, SparseCategoricalCrossEntropy, Hinge, LogCosh, SquaredHinge, CosineSimilarityLoss, BinaryFocalLoss, CategoricalFocalLoss, Poisson
from .layers import Dense, Conv2D, Flatten, MaxPool2D, BatchNorm2D, BatchNorm1D
from .optimisers import Optimiser, SGD, Adam, Momentum, Nesterov, AdaGrad, RMSProp, AdamW, Nadam, Adamax, AMSGrad, Lion 
from .metrics import Metrics, CategoricalAccuracy, BinaryAccuracy, MeanSE, MeanAE, Precision, Recall, F1Score, ConfusionMatrix, TopKCatergoicalAccuracy, RootMeanSquaredError, MeanAbsolutePercentageError, R2Score



class NeuralNetwork():
    def __init__(self, layers, batch_size=32, Loss_Algorithm="", Optimiser="", Metrics="", Regulariser=None):
        self.layers = layers
        self.Loss_Algorithm = Loss_Algorithm
        self.Optimiser = Optimiser
        self.batch_size = batch_size
        self.metrics = Metrics
        self.Regulariser = Regulariser

    def forward(self, inputs, training=True):
        for layer in self.layers:
            inputs = layer.forward(inputs, training)
        
        return inputs

    def backward(self, outputs):
        for layer in reversed(self.layers):
            skip_softmax = isinstance(layer, Softmax) and isinstance(self.Loss_Algorithm, (CrossEntropyLoss, SparseCategoricalCrossEntropy))
            skip_sigmoid = isinstance(layer, Sigmoid) and isinstance(self.Loss_Algorithm, BinaryCrossEntropy)

            if skip_sigmoid or skip_softmax:
                continue
            else:
                outputs = layer.backward(outputs)
                if isinstance(layer, Dense) and self.Regulariser is not None :
                    layer.weight_gradients += self.Regulariser.gradient(layer.weights)

    def train(self, x_input, y_input, epochs, normalise=True, shuffle=True, validation_data=None, CallBacks=[]):

        start = time.perf_counter()
        self.normalise = normalise
        self.callbacks = CallBacks
        self.stop_training = False
        self.epochs = epochs

        if validation_data is not None:
            self.validation_data = validation_data
            self.validation = True
        else:
            self.validation = False

        self._validate_model()

        if x_input is None or y_input is None:
            raise ValueError("The dataset is empty!")
        
        if len(x_input) != len(y_input):
            raise ValueError("Data sets must be the same size!")
        
        for callback in self.callbacks:
            callback.on_train_begin(self)
        
        if not self.stop_training:
            for epoch in range(epochs):
                self.epoch_logs = {}

                self.total_batches = math.ceil(len(x_input) / self.batch_size)

                for callback in self.callbacks:
                    callback.on_epoch_begin(self, epoch)

                if shuffle:
                    self.indicies = np.random.permutation(len(x_input))
                else:
                    self.indicies = list(range(len(x_input)))

                epoch_loss = 0
                batch_count = 0

                for i in range(len(self.metrics)):
                    self.metrics[i].reset()

                for i in range(0, len(x_input), self.batch_size):
                    self.batch_logs = {}

                    for callback in self.callbacks:
                        callback.on_batch_begin(self, batch_count)

                    end = i + self.batch_size
                    self.data = x_input[self.indicies[i:end]]

                    if not any(isinstance(layer, Conv2D) for layer in self.layers):
                        self.data = self.data.reshape(self.data.shape[0], -1)

                    if not isinstance(self.Loss_Algorithm, SparseCategoricalCrossEntropy):
                        target = self.one_hot(y_input[self.indicies[i:end]])
                    else:
                        target = y_input[self.indicies[i:end]]

                    if self.normalise:
                        self.data = self.data / 255

                    prediction = self.forward(self.data, training=True)
                    loss = self.Loss_Algorithm.forward(prediction, target)

                    if self.Regulariser != None:
                        for layer in self.layers:
                            if isinstance(layer, Dense):
                                loss += self.Regulariser.calculate(layer.weights)

                    self.batch_logs["loss"] = loss

                    for i in range(len(self.metrics)):
                        self.metrics[i].update(prediction, target)

                    epoch_loss += loss
                    dC_da = self.Loss_Algorithm.backward()
                    self.backward(dC_da)

                    self.Optimiser.step(self.layers)

                    batch_count += 1

                    for callback in self.callbacks:
                        callback.on_batch_end(self, batch_count, self.batch_logs)
                
                average_loss = epoch_loss / batch_count

                self.epoch_logs["loss"] = average_loss

                metric_string = []

                for metric in self.metrics:
                    if not metric.is_matrix:
                        value = metric.calculate()

                        metric_string.append(
                            f"{metric.name}: {value:{metric.display_format}}"
                        )

                        self.epoch_logs[metric.name] = value
                

                
                print(f"Train  -  Loss: {average_loss:.7f} | " + " | ".join(metric_string))

                if self.validation:
                    self.test(self.validation_data[0], self.validation_data[1], test=False)

                for callback in self.callbacks:
                    callback.on_epoch_end(self, epoch, self.epoch_logs)
                
                if self.stop_training:
                    print("Early Stopping Triggered!")
                    break
            

            end = time.perf_counter()
            print("Training Complete!")
            print(f"Training took {(end - start):.2f}s")
        
            for callback in self.callbacks:
                callback.on_train_end(self)


    def test(self, x_test, y_test, normalise=True, test=True):
        total_loss = 0
        batch_count = 0


        for i in range(len(self.metrics)):
            self.metrics[i].reset()

        for i in range(0, len(x_test), self.batch_size):
            end = i + self.batch_size
            data = x_test[i:end]

            if not any(isinstance(layer, Conv2D) for layer in self.layers):
                data = data.reshape(data.shape[0], -1)

            if not isinstance(self.Loss_Algorithm, SparseCategoricalCrossEntropy):
                target = self.one_hot(y_test[i:end])
            else:
                target = y_test[i:end]

            if self.normalise:
                data = data / 255
            
            prediction = self.forward(data, training=False)
            loss = self.Loss_Algorithm.forward(prediction, target)

            for i in range(len(self.metrics)):
                self.metrics[i].update(prediction, target)

            total_loss += loss            
            batch_count += 1
            
        average_loss = total_loss / batch_count

        metric_string = []

        self.epoch_logs["val_loss"] = average_loss

        for metric in self.metrics:
            if not metric.is_matrix:
                value = metric.calculate()

                metric_string.append(
                    f"{metric.name}: {value:{metric.display_format}}"
                )

                self.epoch_logs[f"val_{metric.name}"] = value
            
        print(f"Validation  -  Loss: {average_loss:.7f} | " + " | ".join(metric_string))

        for metric in self.metrics:
            if metric.is_matrix:
                print(f"\n{metric.name}")
                print(metric.calculate())
            

    def save(self, filename):
        array = {}

        dense_count = 0
        conv_count = 0

        for layer in self.layers:
            if isinstance(layer, Dense):
                array[f"dense_w{dense_count}"] = layer.weights
                array[f"dense_b{dense_count}"] = layer.biases
                dense_count += 1
            
            elif isinstance(layer, Conv2D):
                array[f"conv_f{conv_count}"] = layer.filters
                array[f"conv_b{conv_count}"] = layer.biases
                conv_count += 1
        
        np.savez(filename, **array)

    def load(self, filename):
        data = np.load(filename)

        dense_count = 0
        conv_count = 0

        for layer in self.layers:
            if isinstance(layer, Dense):
                layer.weights = data[f"dense_w{dense_count}"]
                layer.biases = data[f"dense_b{dense_count}"]
                dense_count += 1
            
            elif isinstance(layer, Conv2D):
                layer.filters = data[f"conv_f{conv_count}"]
                layer.biases = data[f"conv_b{conv_count}"]
                conv_count += 1


    def one_hot(self, target):
        lastDense = 0
        for layer in reversed(self.layers):
            if isinstance(layer, Dense):
                lastDense = layer
                break

        output_size = lastDense.output_size

        if output_size == 1:
            return np.array([target])
        
        one_hot_target = []

        for i in range(len(target)):
            default = np.zeros(output_size)
            default[target[i]] = 1
            one_hot_target.append(default)
        
        return np.array(one_hot_target)

    def _validate_model(self):
        valid_metrics = (CategoricalAccuracy, BinaryAccuracy, MeanSE, MeanAE, Precision, Recall, F1Score, ConfusionMatrix, TopKCatergoicalAccuracy, RootMeanSquaredError, MeanAbsolutePercentageError, R2Score)
        valid_optimisers = (SGD, Adam, Momentum, Nesterov, AdaGrad, RMSProp, AdamW, Nadam, Adamax, AMSGrad, Lion)
        valid_Loss = (MSE, CrossEntropyLoss, BinaryCrossEntropy, MAE, Huber, KL_Divergence, SparseCategoricalCrossEntropy, Hinge, LogCosh, SquaredHinge, CosineSimilarityLoss, BinaryFocalLoss, CategoricalFocalLoss, Poisson)

        restricted_pairs = {
            CrossEntropyLoss: (Softmax),
            CategoricalFocalLoss: (Softmax),
            BinaryCrossEntropy: (Sigmoid),
            BinaryFocalLoss: (Sigmoid), 
            Hinge: (Linear, Tanh),
            SquaredHinge: (Linear, Tanh),
            Poisson: (Softplus, ReLU)

        }

        loss_type = type(self.Loss_Algorithm)

        if loss_type in restricted_pairs:
            if not isinstance(self.layers[-1], restricted_pairs[loss_type]):
                print(f"Ensure {loss_type} goes with one of these: {restricted_pairs[loss_type]}")

        if len(self.layers) == 0:
            raise ValueError("Ensure layers is not empty!")

        for layer in self.layers:
            if not isinstance(layer, Layer):
                raise ValueError(f"{layer} is not a valid Layer Object!")
        
        for i in range(len(self.metrics)):
            if not isinstance(self.metrics[i], valid_metrics):
                raise ValueError("Ensure your Metrics are Correct!")
        
        if len(self.metrics) == 0:
            print("WARNING - Recommend to have at least 1 metrics for training!")
        
        if not isinstance(self.Loss_Algorithm, valid_Loss):
            raise ValueError("Ensure your Loss Algorithm is Correct!")
        
        if not isinstance(self.Optimiser, valid_optimisers):
            raise ValueError("Ensure your Optimiser is Correct!")
        
        previous_dense = 0

        for layer in self.layers:
            if isinstance(layer, Dense):
                if previous_dense != 0:
                    if previous_dense.output_size != layer.input_size:
                        raise ValueError("Ensure the output size of Dense layers equal the input size of the next dense layer!")
                    else:
                        previous_dense = layer
                else:
                    previous_dense = layer
        
        if isinstance(self.layers[-1], Softmax) and not isinstance(self.Loss_Algorithm, CrossEntropyLoss):
            raise ValueError("Softmax must currently be used with CrossEntropyLoss.")
        
        if (not isinstance(self.layers[-1], Softmax) and isinstance(self.Loss_Algorithm, CrossEntropyLoss)):
            raise ValueError("Ensure Cross Entropy Loss uses the Softmax activation function!")
        
        if (not isinstance(self.layers[-1], Sigmoid) and isinstance(self.Loss_Algorithm, BinaryCrossEntropy)):
            raise ValueError("Ensure Binary Cross Entropy ONLY goes with sigmoid.")