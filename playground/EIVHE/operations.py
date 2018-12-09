from playground.EIVHE.math_helper import matrix_to_vector, vector_to_matrix, check_is_matrix
from playground.EIVHE.type_check import check_is_vector
import numpy as np


# c' = c1 + c2
def secure_add_vectors(c1, c2):
    check_is_vector(c1)
    check_is_vector(c2)
    return np.add(c1, c2)


# sc = wx + e
# (Gs)c = wGx + Ge
# Gsc = s'c'

# client compute M.
# g is not encoded. Client encode g using t
def secure_linear_transform_client(encryption_core, g, s, t):
    return encryption_core.key_switching_get_switching_matrix(g.dot(s), t)


# server compute c' = Mc
# server knows w (private key)
def secure_linear_transform_server(encryption_core, c, m):
    return encryption_core.key_switching_get_cipher_from_switching_matrix(c, m)


# vec(s1 H s2) [c1c2/w] = w (x1hx2) + e
# h is not encoded. Client encode h using t
def secure_inner_product_client(encryption_core, s1, s2, h, t):
    s = vector_to_matrix(matrix_to_vector(s1.T.dot(h).dot(s2)), 1, -1)
    switching_matrix = encryption_core.key_switching_get_switching_matrix(s, t)
    return switching_matrix


# server computes inner product
# server knows w (private key)
def secure_inner_product_server(encryption_core, c1, c2, m, w):
    c1_dot_c2 = np.matrix(c1.reshape(-1, 1).dot(c2.reshape(1, -1)))
    vec_c1_dot_c2 = matrix_to_vector(c1_dot_c2)
    c = (vec_c1_dot_c2 / w).round().astype('int64')
    return encryption_core.key_switching_get_cipher_from_switching_matrix(c, m)
