import unittest

from playground.EIVHE.eivhe_helper import *

s = np.matrix([[10, 0, 0, 0], [0, 10, 0, 0], [0, 0, 10, 0], [0, 0, 0, 10]])
c = np.array([1, 2, 3, 4])
w = np.int64(10)
x = np.array([1, 2, 3, 4])


class TestEncryptionMethods(unittest.TestCase):

    def test_decrypt(self):
        np.testing.assert_equal(decrypt(s, c, w), x)

    def test_int_to_bit(self):
        np.testing.assert_equal(int_to_bit(np.int64(1), 3), np.array([0, 0, 1]))
        np.testing.assert_equal(int_to_bit(np.int64(-1), 3), np.array([0, 0, -1]))
        np.testing.assert_equal(int_to_bit(np.int64(2), 3), np.array([0, 1, 0]))
        np.testing.assert_equal(int_to_bit(np.int64(-2), 3), np.array([0, -1, 0]))
        with self.assertRaises(ValueError):
            int_to_bit(np.int64(2), 1)

    def test_compute_c_star(self):
        np.testing.assert_equal(
            compute_c_star(np.array([1, -3]).astype('int64'), 3),
            np.array([0, 0, 1, 0, -1, -1]))

    def test_compute_s_star(self):
        np.testing.assert_equal(
            compute_s_star(np.matrix([[1, 2], [5, 4]]).astype('int64'), 3),
            np.array([[4, 2, 1, 8, 4, 2], [20, 10, 5, 16, 8, 4]])
        )

    def test_sc_scstar(self):
        s = np.matrix([[1, 2], [5, 4]]).astype('int64')
        c = np.array([1, -3]).astype('int64')
        s_star = compute_s_star(s, 3)
        c_star = compute_c_star(c, 3)
        np.testing.assert_equal(
            s.dot(c),
            s_star.dot(c_star))


if __name__ == '__main__':
    unittest.main()
