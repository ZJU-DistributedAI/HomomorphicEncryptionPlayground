import numpy as np

from EIVHE.math_helper import exponential
from EIVHE.safe_nn.layer import Layer


class SafeLinearLayer(Layer):
    def __init__(self, encryption, n_in, n_out):
        self.enc = encryption
        self.c_w = self.enc.encrypt_matrix(Layer.xavier_activation((n_in, n_out)))
        self.c_b = self.enc.encrypt_vector(Layer.xavier_activation((n_out,)))

    def simple_forward(self, x):
        w = self.enc.decrypt_matrix(self.c_w)
        b = self.enc.decrypt_vector(self.c_b)
        return x.dot(w) + b

    def forward(self, c_x):
        print('performing linear forward')
        x_dot_w = self.enc.outer_product(c_x, self.c_w)
        return np.array([self.enc.add(row, self.c_b) for row in x_dot_w])

    def backward(self, learning_rate, _, c_x, *arg):
        output_grad = arg[0]
        input_grad = self.enc.outer_product(output_grad, self.enc.transpose(self.c_w))
        print('calculating dw')
        c_dw = self.enc.outer_product(self.enc.transpose(c_x), output_grad)
        print('calculating db')
        c_db = np.sum(output_grad, axis=0) # I am lazy here....
        self.c_w = self.enc.subtract(self.c_w, self.enc.multiply_scalar(c_dw, learning_rate))
        self.c_b = self.enc.subtract(self.c_b, self.enc.multiply_scalar(c_db, learning_rate))
        return input_grad


# https://stats.stackexchange.com/questions/235528/backpropagation-with-softmax-cross-entropy
class SafeSoftmaxOutputLayer(Layer):
    def __init__(self, encryption):
        self.enc = encryption

    def simple_forward(self, x):
        return exponential(x) / np.sum(exponential(x), axis=1, keepdims=True)

    def forward(self, c_xs):
        print('performing softmax forward')
        c_xs = np.array(c_xs)
        return np.array([self.enc.softmax(c_x) for c_x in c_xs])

    def backward(self, learning_rate, c_y, _, *arg):
        print('performing softmax backward')
        c_t = arg[0]
        return self.enc.subtract(c_y, c_t)
