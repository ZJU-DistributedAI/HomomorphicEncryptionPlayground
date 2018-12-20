import numpy as np

from EIVHE.math_helper import exponential
from EIVHE.safe_nn.layer import Layer


class SafeLinearLayer(Layer):
    def __init__(self, encryption, n_in, n_out):
        self.enc = encryption
        self.w = Layer.xavier_activation((n_in, n_out))
        self.b = Layer.xavier_activation((n_out,))

    def simple_forward(self, x):
        return x.dot(self.w) + self.b

    def forward(self, c_x):
        print('performing linear forward')
        c_b = self.enc.encrypt_vector(self.b)
        dot_product = self.enc.linear_transform(self.w.T, c_x)
        return self.enc.add(dot_product, c_b)

    def backward(self, learning_rate, _, c_x, *arg):
        print('performing linear backward')
        c_output_grad = np.array(arg[0])  # (11,)
        x = self.enc.decrypt_vector(c_x)
        output_grad = self.enc.decrypt_vector(c_output_grad)
        dw = x.reshape(-1, 1).dot(output_grad.reshape(1, -1))
        db = np.sum(output_grad)
        self.w = self.w - learning_rate * dw
        self.b = self.b - learning_rate * db

        c_input_grad = self.enc.linear_transform(self.w, c_output_grad)
        return c_input_grad  # (785,)


# https://stats.stackexchange.com/questions/235528/backpropagation-with-softmax-cross-entropy
class SafeSoftmaxOutputLayer(Layer):
    def __init__(self, encryption):
        self.enc = encryption

    def simple_forward(self, x):
        return exponential(x) / np.sum(exponential(x))

    def forward(self, c_x):
        print('performing softmax forward')
        return self.enc.softmax(c_x)

    def backward(self, learning_rate, c_y, _, *arg):
        print('performing softmax backward')
        c_t = arg[0]
        return self.enc.subtract(c_y, c_t)
