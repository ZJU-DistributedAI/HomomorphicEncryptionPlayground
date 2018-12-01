import unittest

from playground.EIVHE.eivhe_helper import *
from playground.EIVHE.operations import secure_add_vectors, secure_linear_transform_client, \
    secure_linear_transform_server, secure_inner_product_client, secure_inner_product_server

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


class TestOperationMethods(unittest.TestCase):
    def setUp(self):
        self.eivhe = EIVHE(100, np.int64(10), np.int64(10))

    @for_examples(
        ([1], [2], 999, [[5]]),
        ([1, 2], [2, 3], 999, [[1, 2], [3, 4]])
    )
    def test_secure_add_vectors(self, x, y, w, t):
        x = np.array(x)
        y = np.array(y)
        w = np.int64(w)
        t = np.matrix(t)
        correct_result = x + y
        print('Testing {} + {}, should equal to {}'.format(x, y, correct_result))
        s0 = self.eivhe.naive_encrypt_secret(w, x)
        x0 = x
        y0 = y
        # Key switching
        s1 = self.eivhe.key_switching_get_secret(t)
        x1 = self.eivhe.key_switching_get_cipher(x0, s0, t)
        y1 = self.eivhe.key_switching_get_cipher(y0, s0, t)
        # Checking sum
        encrypted_result = secure_add_vectors(x1, y1)
        result = self.eivhe.decrypt(s1, encrypted_result, w)
        np.testing.assert_equal(result, correct_result)

    @for_examples(
        ([1], [[1]], 999, [[5]]),
        ([2, 3], [[1, 2], [3, 4]], 999, [[1], [2]])
    )
    def test_secure_linear_transform(self, x, g, w, t):
        x = np.array(x)
        g = np.matrix(g)
        w = np.int64(w)
        t = np.matrix(t)
        correct_result = np.squeeze(np.array(g.dot(x)))
        print('Testing {} linear transform {}, should equal to {}'.format(g, x, correct_result))
        # Encrypt x
        s0 = self.eivhe.naive_encrypt_secret(w, x)
        x0 = x
        # Key switching for x
        s1 = self.eivhe.key_switching_get_secret(t)
        x1 = self.eivhe.key_switching_get_cipher(x0, s0, t)
        # On client side (g is not encrypted!)
        m = secure_linear_transform_client(self.eivhe, g, s1, t)
        # On server side
        encrypted_result = secure_linear_transform_server(self.eivhe, x1, m)
        # Checking linear transform
        result = self.eivhe.decrypt(s1, encrypted_result, w)
        np.testing.assert_equal(result, correct_result)

    @for_examples(
        ([1], [1], [[1]], 999, [[7]], [[8]], [[9]]),
        ([1], [1, 2], [[1, 2]], 999, [[5]], [[5], [6]], [[9]])
    )
    def test_secure_weighted_inner_product(self, x1, x2, h, w, t1, t2, t3):
        x1 = np.array(x1)
        x2 = np.array(x2)
        h = np.matrix(h)
        w = np.int64(w)
        t1 = np.matrix(t1)
        t2 = np.matrix(t2)
        t3 = np.matrix(t3)
        correct_result = x1.dot(h).dot(x2)
        print(
            'Testing weighted dot product x1: {}, h: {}, x2: {}, should equal to: {}'.format(x1, h, x2, correct_result))
        # Encrypt x1, x2
        x1s0 = self.eivhe.naive_encrypt_secret(w, x1)
        x1c0 = x1
        x2s0 = self.eivhe.naive_encrypt_secret(w, x2)
        x2c0 = x2
        # Key switching for x1 x2
        s1 = self.eivhe.key_switching_get_secret(t1)
        s2 = self.eivhe.key_switching_get_secret(t2)
        x1c1 = self.eivhe.key_switching_get_cipher(x1c0, x1s0, t1)
        x2c1 = self.eivhe.key_switching_get_cipher(x2c0, x2s0, t2)
        # On client side (h is not encrypted)
        m = secure_inner_product_client(self.eivhe, s1, s2, h, t3)
        s3 = self.eivhe.key_switching_get_secret(t3)
        # On sever
        encrypted_result = secure_inner_product_server(self.eivhe, x1c1, x2c1, m, w)
        # Checking weighted inner product
        result = self.eivhe.decrypt(s3, encrypted_result, w)
        np.testing.assert_equal(result, correct_result)


if __name__ == '__main__':
    unittest.main()
