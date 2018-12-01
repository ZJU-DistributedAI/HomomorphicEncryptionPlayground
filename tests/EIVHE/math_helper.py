import unittest

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

    def test_vector_to_matrix(self):
        np.testing.assert_equal(
            vector_to_matrix(np.array([1, 2, 3, 4, 5, 6]), 2, 3),
            np.matrix([[1, 2, 3], [4, 5, 6]])
        )

    def test_generate_random_matrix(self):
        matrix_generated = generate_random_matrix(2, 3, 2)
        self.assertEqual(matrix_generated.shape, (2, 3))
        self.assertEqual(np.max(matrix_generated), 1)
        self.assertTrue(isinstance(matrix_generated, np.matrix))


if __name__ == '__main__':
    unittest.main()
