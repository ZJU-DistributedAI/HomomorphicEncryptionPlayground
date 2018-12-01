import time

from playground.EIVHE.math_helper import *
from playground.EIVHE.type_check import *
import numpy as np


# sc = wx + e, where:
# x: plain text vector [Z^m]
# s: secret key matrix [Z^(m x n)], and it was assumed that |s| << w
#       (w is much greater than any element of s)
# c: cipher text vector [Z^n]
# w: a large integer
# e: an error term with elements smaller than w/2


# with knowledge of s(secret), decryption is straight forward
# x = (sc - e)/w = sc/w - e/w = round(sc/w)
class EIVHE:

    def __init__(self, number_of_bits, a_bound, e_bound):
        check_is_int(number_of_bits)
        check_is_int64(a_bound)
        check_is_int64(e_bound)
        self.number_of_bits = number_of_bits
        self.a_bound = a_bound
        self.e_bound = e_bound
        self.times_by = np.array([2 ** (self.number_of_bits - x - 1) for x in range(self.number_of_bits)])
        self.integer_to_bit_map = {}
        for i in range(-2 ** number_of_bits + 1, 2 ** number_of_bits):
            self.integer_to_bit_map[i] = self._int_to_bit(np.int64(i))

    # key switching:
    # we would like to find s'c' = sc
    # there are two steps:

    # step 1:
    # converting c and s into an intermediate bit representation c* and s*
    # here we introduce l, which is the length of the bit vector

    def _int_to_bit(self, integer):
        integer_copy = integer
        check_is_int64(integer)
        is_negative = integer < 0
        integer = np.abs(integer)
        result = list()
        while integer > 0:
            result.append(int(integer % 2))
            integer = int(integer / 2)
        if self.number_of_bits < len(result):
            raise ValueError("l too small for binary integer {}, need {}".format(integer_copy, len(result)))
        result.extend(np.zeros(self.number_of_bits - len(result)))
        result = np.array(list(reversed(result))).astype('int64')
        if is_negative:
            return -result
        else:
            return result

    def _compute_c_star(self, c):
        check_is_vector(c)
        result = np.array([self.integer_to_bit_map[x] for x in c])
        return np.array(np.resize(result, (result.size,)))

    def _compute_s_star(self, s):
        check_is_matrix(s)
        result = np.array([self.times_by * x for x in np.nditer(s)])
        return np.matrix(np.resize(result, (s.shape[0], s.shape[1] * self.number_of_bits)))

    # step 2:
    # switching the bit representation into the desired secret key

    # m: key-switching matrix   [Z^(n' x nl)]
    # which satisfies S'M = S* + E   [Z^(m x nl)]
    def _compute_switching_matrix_m(self, s_star, t):
        check_is_matrix(s_star)
        check_is_matrix(t)
        if t.shape[0] != s_star.shape[0]:
            raise ValueError("Dimension does not match, most likely something wrong with t")
        a = generate_random_matrix(t.shape[1], s_star.shape[1], self.a_bound)
        e = generate_random_matrix(s_star.shape[0], s_star.shape[1], self.e_bound)
        result = vertical_cat(s_star - t.dot(a) + e, a)
        return result

    # s' = [I, t]       [Z^(m x n')]
    @staticmethod
    def _compute_new_s(t):
        check_is_matrix(t)
        return horizontal_cat(np.matrix(np.eye(len(t))), t)

    # c' = Mc*
    @staticmethod
    def _compute_new_c(m, c_star):
        check_is_matrix(m)
        check_is_vector(c_star)
        return np.squeeze(np.array(m.dot(c_star))).astype('int64')

    # Encryption & Decryption

    # sc = wx + e
    # (wI)x = wx
    @staticmethod
    def naive_encrypt_secret(w, plain_text_size):
        return np.matrix(np.eye(plain_text_size) * w)

    # This method is used in operations
    def key_switching_get_switching_matrix(self, s, t):
        s_star = self._compute_s_star(s)
        switching_matrix = self._compute_switching_matrix_m(s_star, t)
        return switching_matrix

    # This method is used in operations
    def key_switching_get_cipher_from_switching_matrix(self, c, m):
        c_star = self._compute_c_star(c)
        return self._compute_new_c(m, c_star)

    def key_switching_get_secret(self, t):
        return self._compute_new_s(t)

    def key_switching_get_cipher(self, c, s, t):
        m = self.key_switching_get_switching_matrix(s, t)
        return self.key_switching_get_cipher_from_switching_matrix(c, m)

    @staticmethod
    def decrypt(s, c, w):
        check_is_matrix(s)
        check_is_vector(c)
        check_is_int64(w)
        return np.squeeze(np.array((s.dot(c) / w).astype('float').round().astype('int64')))
