import collections

from playground.EIVHE.math_helper import generate_random_matrix, vertical_cat, horizontal_cat
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


# key switching:
# we would like to find s'c' = sc
# there are two steps:

# step 1:
# converting c and s into an intermediate bit representation c* and s*
# here we introduce l, which is the length of the bit vector

def int_to_bit(integer, l):
    check_is_int64(integer)
    check_is_int(l)

    is_negative = integer < 0
    integer = np.abs(integer)
    result = list()
    while integer > 0:
        result.append(int(integer % 2))
        integer = int(integer / 2)
    if l < len(result):
        raise ValueError("l too small for binary integer")
    result.extend(np.zeros(l - len(result)))
    result = np.array(list(reversed(result))).astype('int64')
    if is_negative:
        return -result
    else:
        return result


def compute_c_star(c, l):
    check_is_vector(c)
    check_is_int(l)
    result = np.array([int_to_bit(x, l) for x in c])
    return np.array(np.resize(result, (result.size,)))


def compute_s_star(s, l):
    check_is_matrix(s)
    check_is_int(l)
    times_by = np.array([2 ** (l - x - 1) for x in range(l)])
    result = np.array([times_by * x for x in np.nditer(s)])
    return np.matrix(np.resize(result, (s.shape[0], s.shape[1] * l)))


# step 2:
# switching the bit representation into the desired secret key


# m: key-switching matrix   [Z^(n' x nl)]
# which satisfies S'M = S* + E   [Z^(m x nl)]
def compute_switching_matrix_m(s_star, t, a_bound, e_bound):
    check_is_matrix(s_star)
    check_is_matrix(t)
    check_is_int64(a_bound)
    check_is_int64(e_bound)
    if t.shape[0] != s_star.shape[0]:
        raise ValueError("Dimension does not match, most likely something wrong with t")
    a = generate_random_matrix(t.shape[1], s_star.shape[1], a_bound)
    e = generate_random_matrix(s_star.shape[0], s_star.shape[1], e_bound)

    return vertical_cat(s_star - t.dot(a) + e, a)


# s' = [I, t]       [Z^(m x n')]
def compute_new_s(t):
    check_is_matrix(t)
    return horizontal_cat(np.matrix(np.eye(len(t))), t)


# c' = Mc*
def compute_new_c(m, c_star):
    check_is_matrix(m)
    check_is_vector(c_star)
    return np.squeeze(np.array(m.dot(c_star))).astype('int64')


SecretCipher = collections.namedtuple('SecretCipher', ['secret', 'cipher'])


# Complete key switching:
def key_switching(secret_cipher, t, length=100, a_bound=np.int64(10), e_bound=np.int64(10)):
    s = secret_cipher.secret
    c = secret_cipher.cipher
    s_star = compute_s_star(s, length)
    c_star = compute_c_star(c, length)
    m = compute_switching_matrix_m(s_star, t, a_bound, e_bound)
    s_new = compute_new_s(t)
    c_new = compute_new_c(m, c_star)
    return SecretCipher(secret=s_new, cipher=c_new)


# sc = wx + e
# (wI)x = wx
def encrypt(x, w, t, length=100, a_bound=np.int64(10), e_bound=np.int64(10)):
    check_is_vector(x)
    s = np.matrix(np.eye(x.size) * w)
    c = x
    secret_cipher = SecretCipher(secret=s, cipher=c)
    return key_switching(secret_cipher, t, length, a_bound, e_bound)


def decrypt(s, c, w):
    check_is_matrix(s)
    check_is_vector(c)
    check_is_int64(w)
    return np.squeeze(np.array((s.dot(c) / w).astype('float').round().astype('int64')))
