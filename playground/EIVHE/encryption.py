import os
import cPickle as pickle
from playground.EIVHE.math_helper import *

# This is the encryption class with optimizations
from playground.EIVHE.operations import *


class Encryption:

    def __init__(self, encryption_core, w, scale, t_bound, input_range):
        self.encryption_core = encryption_core
        self.w = w
        self.scale = np.int64(scale)
        self.t_bound = t_bound
        self.input_range = input_range
        self.t_cache = self.load_t_cache()
        self.s_cache = self.load_s_cache()

    # Note that here, t is always with 1 row. hence I is always 1 for S' = [I, T].
    # Cipher is always one dimension higher than plain text
    def load_t_cache(self):
        t_cache = [generate_random_matrix(x, 1, self.t_bound) for x in range(self.w)]
        return t_cache

    # Secret is always of the size (plaintext + 1)
    def load_s_cache(self):
        s_cache = [self.encryption_core.key_switching_get_secret(t) for t in self.t_cache]
        return s_cache

    def get_t(self, size):
        if size >= self.input_range:
            raise ValueError("size {} exceeded input range {}".format(size, self.input_range))
        else:
            return self.t_cache[size]

    def get_s(self, size):
        if size >= self.input_range:
            raise ValueError("size {} exceeded input range {}".format(size, self.input_range))
        else:
            return self.s_cache[size]

    def encrypt_vector(self, vector):
        vector = np.multiply(np.array(vector), self.scale).round().astype(np.int64)
        s0 = self.encryption_core.naive_encrypt_secret(self.w, vector.size)
        c0 = vector
        t1 = self.get_t(vector.size)
        c1 = self.encryption_core.key_switching_get_cipher(c0, s0, t1)
        return c1

    def decrypt_vector(self, cipher):
        secret = self.get_s(len(cipher) - 1)
        result = self.encryption_core.decrypt(secret, cipher, self.w)
        return np.array(result / np.float64(self.scale))

    # Do not need to scale because it is calling encrypt vector
    def encrypt_number(self, number):
        x = np.array([number])
        cipher = self.encrypt_vector(x)
        return cipher

    def decrypt_number(self, cipher):
        result = self.encryption_core.decrypt(self.get_s(1), cipher, self.w)
        return result / np.float64(self.scale)

    # Encode each row of the matrix
    def encrypt_matrix(self, matrix):
        matrix = np.multiply(np.matrix(matrix), self.scale).round().astype(np.int64)
        column_size = matrix.shape[1]
        s0 = self.encryption_core.naive_encrypt_secret(self.w, column_size)
        t1 = self.get_t(column_size)
        encrypted_matrix = np.matrix([
            self.encryption_core.key_switching_get_cipher(np.array(x).reshape(-1), s0, t1)
            for x in matrix
        ])
        return encrypted_matrix

    def decrypt_matrix(self, cipher):
        result = self.encryption_core.decrypt(self.get_s(cipher.shape[1] - 1), cipher, self.w)
        return np.matrix(result / np.float64(self.scale))

    @staticmethod
    def add(cipher1, cipher2):
        assert (cipher1.shape == cipher2.shape)
        return np.add(np.array(cipher1), np.array(cipher2))

    @staticmethod
    def subtract(cipher1, cipher2):
        assert (cipher1.shape == cipher2.shape)
        return np.subtract(np.array(cipher1), np.array(cipher2))

    # Be very careful here, calling round will lose precision in multiply
    @staticmethod
    def multiply_scalar(cipher, number):
        return np.array(np.multiply(cipher, np.float64(number))).round().astype(np.int64)

    # Be very careful here, calling round will lose precision in divide
    @staticmethod
    def divide_scalar(cipher, number):
        return np.array(np.divide(cipher, np.float64(number))).round().astype(np.int64)

    def weighted_inner_product(self, cipher1, h, cipher2):
        # On Client, note that result always has dimension 1.
        secret1 = self.get_s(len(cipher1) - 1)
        secret2 = self.get_s(len(cipher2) - 1)
        m = secure_inner_product_client(self.encryption_core, secret1, secret2, h, self.get_t(1))
        # On Server
        cipher = secure_inner_product_server(self.encryption_core, cipher1, cipher2, m, self.w)
        return self.divide_scalar(cipher, self.scale)

    def linear_transform(self, g, cipher):
        # On Client
        gt = self.get_t(g.shape[0])
        gs = self.encryption_core.key_switching_get_secret(gt)
        input_secret = self.get_s(len(cipher) - 1)
        m = secure_linear_transform_client(self.encryption_core, g, input_secret, gt)
        # On Server
        gt_cipher = secure_linear_transform_server(self.encryption_core, cipher, m)
        # Perform key switching again to switch back to keys in get_t
        t = self.get_t(g.shape[0])
        cipher = self.encryption_core.key_switching_get_cipher(gt_cipher, gs, t)
        return cipher

    @staticmethod
    def one_hot_transform(number_cipher, total_elements, element_index):
        result = np.append(np.zeros(element_index), number_cipher)
        result = np.append(result, np.zeros(total_elements - element_index - 1))
        return result

    def transpose(self, encrypted_matrix):
        encrypted_matrix = np.array(encrypted_matrix)
        n_rows = encrypted_matrix.shape[0]
        n_cols = encrypted_matrix.shape[1] - 1
        eye_n_cols = np.eye(n_cols)
        encrypted_eye_n_cols = np.array(self.encrypt_matrix(eye_n_cols))
        transpose = []
        for col in range(n_cols):
            new_row_after_transpose = np.zeros(n_rows + 1)
            encrypted_one_hot_n_col = encrypted_eye_n_cols[col]
            for row in range(n_rows):
                cipher_scalar = self.weighted_inner_product(encrypted_matrix[row], eye_n_cols, encrypted_one_hot_n_col)
                new_row_after_transpose += self.one_hot_transform(cipher_scalar, n_rows, row)
            transpose.append(new_row_after_transpose)
        return np.array(transpose)

    # 1. + x + x*x/2. + x*x*x/6.
    # Note: losing precision here
    def exponential(self, x):
        one = self.encrypt_number(1)
        x_power_2 = self.weighted_inner_product(x, np.eye(1), x)
        x_power_3 = self.weighted_inner_product(x_power_2, np.eye(1), x)
        result = one
        result = self.add(result, x)
        result = self.add(result, self.divide_scalar(x_power_2, 2))
        result = self.add(result, self.divide_scalar(x_power_3, 6))
        return result

    def cipher_list_to_cipher_vector(self, cipher_list):
        number_of_elements = len(cipher_list)
        eye = np.array(np.eye(number_of_elements))
        result = np.zeros(number_of_elements + 1)
        for i in range(number_of_elements):
            result += self.one_hot_transform(cipher_list[i], number_of_elements, i)
        return result

    def exponential_vector(self, vector):
        number_of_elements = vector.size - 1
        eye = np.array(np.eye(number_of_elements))
        cipher_list = []
        eye_encrypted = np.array(self.encrypt_matrix(eye))
        for i in range(number_of_elements):
            cipher_scalar = self.weighted_inner_product(vector, eye, eye_encrypted[i])
            cipher_exponential = self.exponential(cipher_scalar)
            cipher_list.append(cipher_exponential)
        return self.cipher_list_to_cipher_vector(cipher_list)

    # Note: losing precision here
    def softmax(self, vector):
        number_of_elements = vector.size - 1
        eye = np.array(np.eye(number_of_elements))
        result = np.zeros(number_of_elements + 1)
        exponential_sum = np.zeros(2)
        eye_encrypted = np.array(self.encrypt_matrix(eye))
        for i in range(number_of_elements):
            cipher_scalar = self.weighted_inner_product(vector, eye, eye_encrypted[i])
            cipher_exponential = self.exponential(cipher_scalar)
            result += self.one_hot_transform(cipher_exponential, number_of_elements, i)
            exponential_sum += cipher_exponential

        one = self.encrypt_number(1)
        scale = np.sum(exponential_sum) / np.sum(one)
        return self.divide_scalar(result, scale)

    def outer_product(self, cipher1, cipher2):
        cipher1 = np.array(cipher1)
        cipher2 = np.array(cipher2)
        eye = np.eye(cipher1.shape[1] - 1)
        cipher2_transpose = np.array(self.transpose(cipher2))
        element_result = [[self.weighted_inner_product(cipher1_row, eye, cipher2_row)
                           for cipher2_row in cipher2_transpose] for cipher1_row in cipher1]
        result = [self.cipher_list_to_cipher_vector(element_row) for element_row in element_result]
        return np.array(result)

    def sum(self, vector_cipher):
        number_of_elements = len(vector_cipher) - 1
        ones = self.encrypt_vector(np.ones(number_of_elements))
        return self.weighted_inner_product(vector_cipher, np.eye(number_of_elements), ones)
