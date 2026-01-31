import torch

class SGD:
    def optim(self, parameters, lr = 0.01):
        for parameter in parameters:
            parameter.data -= lr * parameter.grad