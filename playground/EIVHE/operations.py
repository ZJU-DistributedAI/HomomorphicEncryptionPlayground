from playground.EIVHE.math_helper import matrix_to_vector, vector_to_matrix
from playground.EIVHE.type_check import check_is_vector
import numpy as np


# c' = c1 + c2
def secure_add_vectors(c1, c2):
    check_is_vector(c1)
    check_is_vector(c2)
    return c1 + c2


# sc = wx + e
# (Gs)c = wGx + Ge
# Gsc = s'c'

# Client compute M.
# Note that this step is performed on client because G is not encrypted
def secure_linear_transform_client(eivhe, g, s, t):
    return eivhe.key_switching_get_switching_matrix(g.dot(s), t)


# Server compute c' = Mc
def secure_linear_transform_server(eivhe, c, m):
    return eivhe.key_switching_get_cipher_from_switching_matrix(c, m)


# vec(s1 H s2) [c1c2/w] = w (x1hx2) + e
def secure_inner_product_client(eivhe, s1, s2, h, t):
    s = vector_to_matrix(matrix_to_vector(s1.T.dot(h).dot(s2)), 1, -1)
    return eivhe.key_switching_get_switching_matrix(s, t)


def secure_inner_product_server(eivhe, c1, c2, m, w):
    vec_c1_dot_c2 = matrix_to_vector(np.matrix(c1.reshape(-1, 1).dot(c2.reshape(1, -1))))
    c = (vec_c1_dot_c2 / w).round().astype('int64')
    return eivhe.key_switching_get_cipher_from_switching_matrix(c, m)