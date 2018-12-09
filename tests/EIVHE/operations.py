import unittest

from playground.EIVHE.encryption_core import *
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
        self.encryption_core = EncryptionCore(15, np.int64(2), np.int64(2))
        self.w = np.int64(1000)

    @for_examples(
        ([1], [2], [[5]]),
        ([1], [2], [[50]]),
        ([1], [2], [[500]]),
        ([1], [2], [[-5]]),
        ([1], [2], [[-50]]),
        ([1], [2], [[-500]]),
        ([1, 2], [2, 3], [[1, 2], [3, 4]]),
        ([1, 2], [2, 3], [[1000, 2000], [3000, 4000]]),
        ([1, 2], [2, 3], [[1, -2], [3, -4]]),
        ([1, 2], [2, 3], [[1000, -2000], [3000, -4000]])
    )
    def test_secure_add_vectors(self, x0, y0, t):
        x0 = np.array(x0)
        y0 = np.array(y0)
        t = np.matrix(t)
        correct_result = x0 + y0
        print('Testing {} + {}, should equal to {}'.format(x0, y0, correct_result))
        print('Using t: {}'.format(t))
        # Key switching
        s0 = self.encryption_core.naive_encrypt_secret(self.w, x0.size)
        x0c = x0
        y0c = y0
        s1 = self.encryption_core.key_switching_get_secret(t)
        x1c = self.encryption_core.key_switching_get_cipher(x0c, s0, t)
        y1c = self.encryption_core.key_switching_get_cipher(y0c, s0, t)
        print('Encrypt x, with shapes - x: {}, t: {}, naive secret: {}, new cipher: {}'.format(
            x0c.shape, t.shape, s0.shape, x1c.shape
        ))
        print('Encrypt y, with shapes - y: {}, t: {}, naive secret: {}, new cipher: {}'.format(
            y0c.shape, t.shape, s0.shape, y1c.shape
        ))
        # Checking sum
        encrypted_result = secure_add_vectors(x1c, y1c)
        result = self.encryption_core.decrypt(s1, encrypted_result, self.w)
        np.testing.assert_equal(result, correct_result)
        print('w: {}, largest integer element in result - encrypted:{}, decrypted:{}'.format(
            self.w, np.max(encrypted_result), np.max(result)
        ))

    @for_examples(
        ([1], [[1]], [[1, 2]], [[4, 5, 6]]),
        # ([1], [[1]], [[1000, 2000]], [[4000, 5000, 6000]]),
        # ([1], [[1]], [[-1000, -2000]], [[-4000, -5000, -6000]]),
        # ([2, 3], [[1, 2], [3, 4], [5, 6]], [[3], [4]], [[5], [6], [7]]),
        # ([2, 3], [[1, 2], [3, 4], [5, 6]], [[3000], [4000]], [[1000], [2000], [3000]]),
        # ([2, 3], [[1, 2], [3, 4], [5, 6]], [[3000], [4000]], [[1000], [2000], [3000]]),
        # ([2, 3], [[1, 2], [3, 4], [5, 6]], [[-3000], [-4000]], [[-1000], [-2000], [-3000]])
    )
    def test_secure_linear_transform(self, x0, g, x0t, gt):
        x0 = np.array(x0)
        g = np.matrix(g)
        x0t = np.matrix(x0t)
        gt = np.matrix(gt)
        correct_result = np.squeeze(np.array(g.dot(x0)))
        print('Testing {} linear transform {}, should equal to {}'.format(g, x0, correct_result))
        print('Using t_x0: {}, t_g: {}'.format(x0t, gt))
        # Encrypt x
        x0s = self.encryption_core.naive_encrypt_secret(self.w, x0.size)
        x0c = x0
        print(
            'x with shape: {}, t with shape: {}, naive secret with shape: {}, '.format(x0.shape, x0t.shape, x0s.shape))
        # Key switching for x
        x1s = self.encryption_core.key_switching_get_secret(x0t)
        x1c = self.encryption_core.key_switching_get_cipher(x0c, x0s, x0t)
        print('Performed key switching for x, where new cipher is of the shape: {}'.format(x1c.shape))
        # On client side (g is not encrypted!)
        gs = self.encryption_core.key_switching_get_secret(gt)
        m = secure_linear_transform_client(self.encryption_core, g, x1s, gt)
        print('Calculating switch matrix(G) on client side, with shapes - g:{}, t:{}, s:{}, m:{}'.format(
            g.shape, gt.shape, x1s.shape, m.shape))
        # On server side
        encrypted_result = secure_linear_transform_server(self.encryption_core, x1c, m)
        print('Final encrypted result with shape: {}, with largest element: {}'.format(
            encrypted_result.shape, np.max(encrypted_result)))
        # Checking linear transform
        result = self.encryption_core.decrypt(gs, encrypted_result, self.w)
        np.testing.assert_equal(result, correct_result)

    @for_examples(
        ([1], [[1]], [1], [[7]], [[9]], [[8]]),
        ([1], [[1, 2]], [1, 2], [[5]], [[5], [6]], [[9]]),
        ([1, 2], [[1, 2, 3], [3, 2, 1]], [3, 2, 1], [[1], [2]], [[2], [3], [2]], [[1000]])
    )
    def test_secure_weighted_inner_product(self, x1, h, x2, x1t, x2t, result_t):
        x1 = np.array(x1)
        x2 = np.array(x2)
        h = np.matrix(h)
        x1t = np.matrix(x1t)
        x2t = np.matrix(x2t)
        result_t = np.matrix(result_t)
        correct_result = x1.dot(h).dot(x2)
        print('Testing weighted dot product x1:{}, h:{}, x2:{}, should equal to:{}'.format(x1, h, x2, correct_result))
        print('Using t - x1t:{}, x2t:{}, ht:{}'.format(x1t, x2t, result_t))
        # Encrypt x1, x2
        x1s0 = self.encryption_core.naive_encrypt_secret(self.w, x1.size)
        x1c0 = x1
        x2s0 = self.encryption_core.naive_encrypt_secret(self.w, x2.size)
        x2c0 = x2
        # Key switching for x1 x2
        x1s = self.encryption_core.key_switching_get_secret(x1t)
        x2s = self.encryption_core.key_switching_get_secret(x2t)
        x1c = self.encryption_core.key_switching_get_cipher(x1c0, x1s0, x1t)
        x2c = self.encryption_core.key_switching_get_cipher(x2c0, x2s0, x2t)
        print('With shapes- x1s:{}, x2s:{}, x1c:{}, x2c{}'.format(x1s.shape, x2s.shape, x1c.shape, x2c.shape))
        # On client side (h is not encrypted)
        m = secure_inner_product_client(self.encryption_core, x1s, x2s, h, result_t)
        result_s = self.encryption_core.key_switching_get_secret(result_t)
        # On sever
        encrypted_result = secure_inner_product_server(self.encryption_core, x1c, x2c, m, self.w)
        # Checking weighted inner product
        result = self.encryption_core.decrypt(result_s, encrypted_result, self.w)
        print('Result: {}'.format(result))
        np.testing.assert_equal(result, correct_result)


if __name__ == '__main__':
    unittest.main()
