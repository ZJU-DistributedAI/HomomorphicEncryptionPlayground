import numpy as np

from playground.EIVHE.math_helper import exponential


def mutate_array(weight, mutation):
    shape = np.shape(weight)
    # scale = np.max(np.absolute(weight))
    mutation = np.random.choice([-1, 1], size=shape, p=[mutation, 1 - mutation])
    return np.multiply(weight, mutation)


class Layer(object):
    def forward(self, x):
        pass

    def explore(self, settings):
        pass

    def mutate(self, settings):
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

    def explore(self, settings):
        sigma = settings['sigma']

        self.w += np.random.normal(0, 1 * sigma, self.w.shape)
        self.b += np.random.normal(0, 1 * sigma, self.b.shape)

    def mutate(self, settings):
        mutation = settings['mutation']
        self.w = mutate_array(self.w, mutation)
        self.b = mutate_array(self.b, mutation)


# https://stats.stackexchange.com/questions/235528/backpropagation-with-softmax-cross-entropy
class SoftmaxOutputLayer(Layer):

    def forward(self, x):
        return exponential(x) / np.sum(exponential(x))

    def explore(self, sigma):
        pass

    def mutate(self, mutation):
        pass


class Relu(Layer):
    def forward(self, x):
        return np.maximum(x, 0)

    def explore(self, sigma):
        pass

    def mutate(self, mutation):
        pass
