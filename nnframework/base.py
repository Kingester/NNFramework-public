class Layer():
    def __init__(self):
        self.trainable = False

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError