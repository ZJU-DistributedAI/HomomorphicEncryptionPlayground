import numpy as np


class Layer(object):
    def simple_forward(self, x):
        pass

    def forward(self, x):
        pass

    def backward(self, learning_rate, y, x, *arg):
        pass

    def explore(self, settings):
        pass

    def mutate(self, settings):
        pass

    @staticmethod
    def xavier_activation(dimension):
        boundary = np.sqrt(6. / sum(list(dimension)))
        return np.random.uniform(-boundary, boundary, dimension)


class HomomorphicEncryptionLayer(Layer):
    def __init__(self, encryption):
        self.enc = encryption

    def simple_forward(self, x):
        return x

    def forward(self, x):
        return self.enc.encrypt_vector(x)

    def backward(self, learning_rate, y, x, *arg):
        return self.enc.decrypt_vector(arg[0])


class HomomorphicDecryptionLayer(Layer):
    def __init__(self, encryption):
        self.enc = encryption

    def simple_forward(self, x):
        return x

    def forward(self, x):
        return self.enc.decrypt_vector(x)

    def backward(self, learning_rate, y, x, *arg):
        return self.enc.encrypt_vector(arg[0])
