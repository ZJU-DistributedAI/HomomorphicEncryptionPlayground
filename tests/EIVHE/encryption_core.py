import unittest

from playground.EIVHE.encryption_core import *


class TestEncryptionMethods(unittest.TestCase):
    def setUp(self):
        self.encryption_core = EncryptionCore(3, np.int64(5), np.int64(5))

    def test_get_binary(self):
        np.testing.assert_equal(self.encryption_core._get_binary(np.int64(1)), np.array([0, 0, 1]))
        np.testing.assert_equal(self.encryption_core._get_binary(np.int64(-1)), np.array([0, 0, -1]))
        np.testing.assert_equal(self.encryption_core._get_binary(np.int64(2)), np.array([0, 1, 0]))
        np.testing.assert_equal(self.encryption_core._get_binary(np.int64(-2)), np.array([0, -1, 0]))
        with self.assertRaises(ValueError):
            self.encryption_core._get_binary(np.int64(9))

    def test_compute_c_star(self):
        np.testing.assert_equal(
            self.encryption_core._compute_c_star(np.array([1, -3]).astype('int64')),
            np.array([0, 0, 1, 0, -1, -1]))

    def test_compute_c_star_for_matrix(self):
        np.testing.assert_equal(
            self.encryption_core._compute_c_star_for_matrix(np.matrix([[1, -3], [-2, 0]]).astype('int64')),
            np.matrix([[0, 0, 1, 0, -1, -1], [0, -1, 0, 0, 0, 0]]))

    def test_compute_s_star(self):
        np.testing.assert_equal(
            self.encryption_core._compute_s_star(np.matrix([[1, 2], [5, 4]]).astype('int64')),
            np.array([[4, 2, 1, 8, 4, 2], [20, 10, 5, 16, 8, 4]])
        )

    def test_sc_equals_sc_star(self):
        s = np.matrix([[1, 2], [5, 4]]).astype('int64')
        c = np.array([1, -3]).astype('int64')
        s_star = self.encryption_core._compute_s_star(s)
        c_star = self.encryption_core._compute_c_star(c)
        np.testing.assert_equal(
            s.dot(c),
            s_star.dot(c_star))

    def test_decrypt(self):
        s = np.matrix([[10, 0, 0, 0], [0, 10, 0, 0], [0, 0, 10, 0], [0, 0, 0, 10]])
        c = np.array([1, 2, 3, 4])
        w = np.int64(10)
        x = np.array([1, 2, 3, 4])
        np.testing.assert_equal(self.encryption_core.decrypt(s, c, w), x)

    def test_encrypt_decrypt_key_switch(self):
        # Pick x and w
        x = np.array([1, 1, 1, 1])
        w = np.int64(1000)
        # Naively create s and c
        s0 = self.encryption_core.naive_encrypt_secret(w, x.size)
        c0 = x
        np.testing.assert_equal(self.encryption_core.decrypt(s0, c0, w), x)
        # Key switching, s0c0 = s1c1
        t1 = np.matrix([[1, 2], [1, 3], [1, 4], [1, 5]])
        s1 = self.encryption_core.key_switching_get_secret(t1)
        c1 = self.encryption_core.key_switching_get_cipher(c0, s0, t1)
        np.testing.assert_equal(self.encryption_core.decrypt(s1, c1, w), x)


if __name__ == '__main__':
    unittest.main()