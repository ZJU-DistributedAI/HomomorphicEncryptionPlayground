import unittest
import numpy as np
from playground.EIVHE.encryption import Encryption
from playground.EIVHE.encryption_core import EncryptionCore

# Magic function to support for_examples decorator
# Refer to https://stackoverflow.com/questions/2798956/python-unittest-generate-multiple-tests-programmatically
__examples__ = "__examples__"


def for_examples(*examples):
    def decorator(f, examples=examples):
        setattr(f, __examples__, getattr(f, __examples__, ()) + examples)
        return f

    return decorator


class TestCaseWithExamplesMetaclass(type):
    def __new__(meta, name, bases, dict):
        def tuplify(x):
            if not isinstance(x, tuple):
                return (x,)
            return x

        for methodname, method in dict.items():
            if hasattr(method, __examples__):
                dict.pop(methodname)
                examples = getattr(method, __examples__)
                delattr(method, __examples__)
                for example in (tuplify(x) for x in examples):
                    def method_for_example(self, method=method, example=example):
                        method(self, *example)

                    methodname_for_example = methodname + "(" + ", ".join(str(v) for v in example) + ")"
                    dict[methodname_for_example] = method_for_example
        return type.__new__(meta, name, bases, dict)


class TestCaseWithExamples(unittest.TestCase):
    __metaclass__ = TestCaseWithExamplesMetaclass
    pass


unittest.TestCase = TestCaseWithExamples


class TestEncryptionSimple(unittest.TestCase):
    def setUp(self):
        number_of_bits = 10
        a_bound = np.int64(5)
        e_bound = np.int64(5)
        encryption_core = EncryptionCore(number_of_bits, a_bound, e_bound)
        t_bound = np.int64(100)
        scale = 100
        w = np.int64(2 ** 10)
        input_range = 1000
        self.encryption = Encryption(encryption_core, w, scale, t_bound, input_range)

    @for_examples(
        [0], [0.01], [1.01, 0.01], [-1.01, -0.01]
    )
    def test_encryption_vector(self, vector):
        cipher = self.encryption.encrypt_vector(vector)
        decrypted = self.encryption.decrypt_vector(cipher)
        print('Test vector encryption:{}, encrypted:{}, decrypted:{}'.format(vector, cipher, decrypted))
        np.testing.assert_equal(decrypted, vector)

    @for_examples(
        0, 0.01, 5.99, -0.01, -5.99
    )
    def test_encryption_number(self, number):
        cipher = self.encryption.encrypt_number(number)
        decrypted = self.encryption.decrypt_number(cipher)
        print('Test integer encryption:{}, encrypted:{}, decrypted:{}'.format(number, cipher, decrypted))
        np.testing.assert_equal(decrypted, number)

    @for_examples(
        [[0]], [[0.01]], [[1.01, 0.01], [-1.01, -0.01]]
    )
    def test_encryption_matrix(self, matrix):
        cipher = self.encryption.encrypt_matrix(matrix)
        decrypted = self.encryption.decrypt_matrix(cipher)
        print('Test matrix encryption:{}, encrypted:{}, decrypted:{}'.format(matrix, cipher, decrypted))
        np.testing.assert_equal(decrypted, matrix)

    @for_examples(
        ([0.02], [0.01]),
        ([0.01, 0.01], [0.02, 0.02])
    )
    def test_vector_add(self, vector1, vector2):
        c1 = self.encryption.encrypt_vector(vector1)
        c2 = self.encryption.encrypt_vector(vector2)
        result = self.encryption.add(c1, c2)
        decrypted = self.encryption.decrypt_vector(result)
        np.testing.assert_equal(decrypted, np.add(vector1, vector2))

    @for_examples(
        ([0.02], [0.01]),
        ([0.01, 0.01], [0.02, 0.02])
    )
    def test_vector_subtract(self, vector1, vector2):
        c1 = self.encryption.encrypt_vector(vector1)
        c2 = self.encryption.encrypt_vector(vector2)
        result = self.encryption.subtract(c1, c2)
        decrypted = self.encryption.decrypt_vector(result)
        np.testing.assert_equal(decrypted, np.subtract(vector1, vector2))

    @for_examples(
        ([0.1, 0.01], 1.),
        ([1, 0.1], 0.1),
    )
    # ([0.19], 0.5)  try this, this will fail due to precision issue
    def test_vector_multiply_scalar(self, vector, number):
        c = self.encryption.encrypt_vector(np.array(vector))
        result = self.encryption.multiply_scalar(c, number)
        decrypted = self.encryption.decrypt_vector(result)
        expected = np.multiply(np.array(vector), number).round(2)
        print('Decrypted: {}, expected: {}'.format(decrypted, expected))
        np.testing.assert_equal(decrypted, expected)

    @for_examples(
        ([0.1, 0.01], 1),
        ([1, 0.1], 2),
        ([0.01], 2)
    )
    def test_vector_divide_scalar(self, vector, number):
        c = self.encryption.encrypt_vector(np.array(vector))
        result = self.encryption.divide_scalar(c, number)
        decrypted = self.encryption.decrypt_vector(result)
        expected = np.divide(np.array(vector), number).round(2)
        print('Decrypted: {}, expected: {}'.format(decrypted, expected))
        np.testing.assert_equal(decrypted, expected)

    @for_examples(
        ([[0.02]], [[0.01]]),
        ([[0.01, -0.01], [-0.01, 0.01]], [[-0.02, 0.02], [0.02, -0.02]])
    )
    def test_matrix_add(self, matrix1, matrix2):
        c1 = self.encryption.encrypt_matrix(matrix1)
        c2 = self.encryption.encrypt_matrix(matrix2)
        result = self.encryption.add(c1, c2)
        decrypted = self.encryption.decrypt_matrix(result)
        np.testing.assert_equal(decrypted, np.add(matrix1, matrix2))

    @for_examples(
        ([[0.02]], [[0.01]]),
        ([[0.01, 0.01]], [[0.02, 0.02]])
    )
    def test_matrix_subtract(self, matrix1, matrix2):
        c1 = self.encryption.encrypt_matrix(matrix1)
        c2 = self.encryption.encrypt_matrix(matrix2)
        result = self.encryption.subtract(c1, c2)
        decrypted = self.encryption.decrypt_matrix(result)
        np.testing.assert_equal(decrypted, np.subtract(matrix1, matrix2))

    @for_examples(
        ([[0.1, 0.01]], 1.),
        ([[1, 0.1]], 0.1),
    )
    # ([0.19], 0.5)  try this, this will fail due to precision issue
    def test_matrix_multiply_scalar(self, matrix, number):
        c = self.encryption.encrypt_matrix(np.array(matrix))
        result = self.encryption.multiply_scalar(c, number)
        decrypted = self.encryption.decrypt_matrix(result)
        expected = np.multiply(np.array(matrix), number).round(decimals=2)
        print('Decrypted: {}, expected: {}'.format(decrypted, expected))
        np.testing.assert_equal(decrypted, expected)

    @for_examples(
        ([[0.1, 0.01]], 1),
        ([[1, 0.1]], 2),
        ([[0.01]], 2)
    )
    def test_matrix_divide_scalar(self, matrix, number):
        c = self.encryption.encrypt_matrix(np.array(matrix))
        result = self.encryption.divide_scalar(c, number)
        decrypted = self.encryption.decrypt_matrix(result)
        expected = np.divide(np.array(matrix), number).round(decimals=2)
        print('Decrypted: {}, expected: {}'.format(decrypted, expected))
        np.testing.assert_equal(decrypted, expected)
