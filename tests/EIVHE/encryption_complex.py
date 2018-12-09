import unittest
import numpy as np
from playground.EIVHE.encryption import Encryption
from playground.EIVHE.encryption_core import EncryptionCore

# Magic function to support for_examples decorator
# Refer to https://stackoverflow.com/questions/2798956/python-unittest-generate-multiple-tests-programmatically
from playground.EIVHE.math_helper import exponential

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


class TestEncryptionComplex(unittest.TestCase):
    def setUp(self):
        number_of_bits = 20
        a_bound = np.int64(5)
        e_bound = np.int64(5)
        encryption_core = EncryptionCore(number_of_bits, a_bound, e_bound)
        t_bound = np.int64(10)
        scale = 100
        w = np.int64(2 ** 10)
        input_range = 1000
        self.encryption = Encryption(encryption_core, w, scale, t_bound, input_range)

    @for_examples(
        ([1.01], [[1]], [0.01]),
        ([0.01, 0.02], [[1, 2, 1], [1, 2, 1]], [1, 2, 1])
    )
    def test_weighted_inner_product(self, vector1, h, vector2):
        expected = np.array(vector1).dot(np.array(h)).dot(np.array(vector2)).round(2)
        c1 = self.encryption.encrypt_vector(vector1)
        c2 = self.encryption.encrypt_vector(vector2)
        print(c1, c2)
        result = self.encryption.weighted_inner_product(c1, h, c2)
        decrypted = self.encryption.decrypt_number(result)
        np.testing.assert_equal(decrypted, expected)

    @for_examples(
        ([[1]], [1]),
        ([[1, 2], [3, 4], [5, 6]], [0.02, 0.03])
    )
    def test_linear_transform(self, g, vector):
        expected = np.squeeze(np.array(g).dot(np.array(vector)))
        c = self.encryption.encrypt_vector(vector)
        result = self.encryption.linear_transform(np.array(g), c)
        decrypted = self.encryption.decrypt_vector(result)
        np.testing.assert_equal(decrypted, expected)

    @for_examples(
        ([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]])
    )
    def test_transpose(self, matrix):
        matrix = np.array(matrix)
        expected = matrix.T
        encrypted = self.encryption.encrypt_matrix(matrix)
        result = self.encryption.transpose(encrypted)
        decrypted = self.encryption.decrypt_matrix(result)
        np.testing.assert_equal(decrypted, expected)
        print(decrypted)

    @for_examples(
        0.01, 0.1, 0
    )
    def test_exponential(self, number):
        expected = round(exponential(number), 2)
        encrypted = self.encryption.encrypt_number(number)
        result = self.encryption.exponential(encrypted)
        decrypted = self.encryption.decrypt_number(result)
        np.testing.assert_almost_equal(decrypted, expected)
        print(decrypted)

    @for_examples(
        [0.01, 0.01, 0.1]
    )
    def test_exponential_vector(self, vector):
        vector = np.array(vector)
        expected = exponential(vector).round(2)
        encrypted = self.encryption.encrypt_vector(vector)
        result = self.encryption.exponential_vector(encrypted)
        decrypted = self.encryption.decrypt_vector(result)
        np.testing.assert_almost_equal(decrypted, expected, decimal=1)

    @for_examples(
        ([0.01, 0.01, 0.01]),
        ([0.01, 0.02, 0.03])
    )
    def test_softmax(self, x):
        x = np.array(x)
        expected = (exponential(x) / np.sum(exponential(x))).round(2)
        encrypted = self.encryption.encrypt_vector(x)
        result = self.encryption.softmax(encrypted)
        decrypted = self.encryption.decrypt_vector(result)
        np.testing.assert_almost_equal(decrypted, expected)
        print(decrypted, expected)

    @for_examples(
        ([[0.1], [0.1]], [[0.1, 0.1]])
    )
    def test_outter_product(self, matrix1, matrix2):
        matrix1 = np.array(matrix1)
        matrix2 = np.array(matrix2)
        expected = matrix1.dot(matrix2)
        encrypted1 = self.encryption.encrypt_matrix(matrix1)
        print(encrypted1)
        encrypted2 = self.encryption.encrypt_matrix(matrix2)
        print(encrypted2)
        result = self.encryption.outer_product(encrypted1, encrypted2)
        decrypted = self.encryption.decrypt_matrix(result)
        np.testing.assert_almost_equal(decrypted, expected)
        print(decrypted, expected)

    @for_examples(
        (0.1, [0.1, 0, 0], 3, 0),
        (0.1, [0, 0.1, 0], 3, 1),
        (0.1, [0, 0, 0.1], 3, 2)
    )
    def test_one_hot_transform(self, original, expected, total_elements, element_index):
        expected = np.array(expected)
        cipher_original = self.encryption.encrypt_number(original)
        transformed = self.encryption.one_hot_transform(cipher_original, total_elements, element_index)
        decrypted = self.encryption.decrypt_vector(transformed)
        np.testing.assert_almost_equal(decrypted, expected)
