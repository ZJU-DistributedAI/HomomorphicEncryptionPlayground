import unittest

from playground.EIVHE.eivhe_helper import encrypt, decrypt, key_switching
from playground.EIVHE.math_helper import *

matrix_a = np.matrix([[1, 1], [1, 1]])
matrix_b = np.matrix([[2, 2], [2, 2]])
matrix_c = np.matrix([[1, 2], [3, 4]])


class TestHelpersMethods(unittest.TestCase):

    def test_horizontal_cat(self):
        np.testing.assert_equal(
            horizontal_cat(matrix_a, matrix_b),
            np.matrix([[1, 1, 2, 2], [1, 1, 2, 2]]))
        with self.assertRaises(ValueError):
            horizontal_cat(np.array([1, 2]), np.array([3, 4])),

    def test_vertical_cat(self):
        np.testing.assert_equal(
            vertical_cat(matrix_a, matrix_b),
            np.matrix([[1, 1], [1, 1], [2, 2], [2, 2]]))

    def test_matrix_to_vector(self):
        np.testing.assert_equal(
            matrix_to_vector(matrix_c),
            np.array([1, 2, 3, 4]))

    def test_generate_random_matrix(self):
        matrix_generated = generate_random_matrix(2, 3, 2)
        self.assertEqual(matrix_generated.shape, (2, 3))
        self.assertEqual(np.max(matrix_generated), 1)
        self.assertTrue(isinstance(matrix_generated, np.matrix))


    def test_homomorphic_encryption(self):
        x = np.array([1,2,3,4])
        w = np.int64(1000)
        t = np.matrix([[4],[4],[4],[4]])
        secret_cipher = encrypt(x, w, t)
        secret_cipher = key_switching(secret_cipher, np.matrix([[5],[5],[4],[5]]))
        print(decrypt(secret_cipher.secret, secret_cipher.cipher, w))



if __name__ == '__main__':
    unittest.main()
