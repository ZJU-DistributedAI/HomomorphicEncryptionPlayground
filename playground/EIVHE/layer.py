import numpy as np


class Layer(object):
    def forward(self, x):
        pass

    def backward(self, learning_rate, y, x, *arg):
        pass

    @staticmethod
    def xavier_activation(dimension):
        boundary = np.sqrt(6. / sum(list(dimension)))
        return np.random.uniform(-boundary, boundary, dimension)


class LinearLayer(Layer):
    def __init__(self, n_in, n_out):
        self.w = Layer.xavier_activation((n_in, n_out))
        self.b = Layer.xavier_activation((n_out,))

    def forward(self, x):
        return x.dot(self.w) + self.b

    def backward(self, learning_rate, y, x, *arg):
        output_grad = arg[0]
        input_grad = output_grad.dot(self.w.T)
        dw = x.T.dot(output_grad)
        db = np.sum(output_grad, axis=0)
        self.w = self.w - learning_rate * dw
        self.b = self.b - learning_rate * db
        return input_grad


class SoftmaxOutputLayer(Layer):
    def forward(self, x):
        x_norm = (x.T - np.max(x, axis=1)).T
        return np.exp(x_norm) / np.sum(np.exp(x_norm), axis=1, keepdims=True)

    def backward(self, learning_rate, y, x, *arg):
        t = arg[0]
        return y - t
