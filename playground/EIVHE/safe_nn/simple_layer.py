import numpy as np

from EIVHE.math_helper import exponential
from EIVHE.safe_nn.layer import Layer


class LinearLayer(Layer):
    def __init__(self, n_in, n_out):
        self.w = Layer.xavier_activation((n_in, n_out))
        self.b = Layer.xavier_activation((n_out,))

    def simple_forward(self, x):
        return self.forward(x)

    def forward(self, x):
        return x.dot(self.w) + self.b

    def backward(self, learning_rate, y, x, *arg):
        output_grad = arg[0]
        input_grad = output_grad.dot(self.w.T)
        dw = x.reshape(-1, 1).dot(output_grad.reshape(1, -1))
        db = np.sum(output_grad)
        self.w = self.w - learning_rate * dw
        self.b = self.b - learning_rate * db
        return input_grad


class ReluLayer(Layer):
    def simple_forward(self, x):
        return self.forward(x)

    def forward(self, x):
        return np.maximum(x, 0.)

    def backward(self, learning_rate, y, x, *arg):
        return np.multiply(np.greater(y, 0.), arg[0])


# https://stats.stackexchange.com/questions/235528/backpropagation-with-softmax-cross-entropy
class SoftmaxOutputLayer(Layer):
    def simple_forward(self, x):
        return self.forward(x)

    def forward(self, x):
        return exponential(x) / np.sum(exponential(x))

    def backward(self, learning_rate, y, x, *arg):
        t = arg[0]
        return y - t
