from playground.EIVHE.type_check import *


# horizontal concat two numpy matrix
def horizontal_cat(matrix_a, matrix_b):
    check_is_matrix(matrix_a)
    check_is_matrix(matrix_b)
    return np.concatenate((matrix_a, matrix_b), 1)


# vertical concat two numpy matrix
def vertical_cat(matrix_a, matrix_b):
    check_is_matrix(matrix_a)
    check_is_matrix(matrix_b)
    return np.concatenate((matrix_a, matrix_b), 0)


# vectorized a matrix
def matrix_to_vector(matrix):
    check_is_matrix(matrix)
    return matrix.getA1()


def vector_to_matrix(vector, row, col):
    check_is_vector(vector)
    return np.matrix(np.reshape(vector, (row, col)))


# Random matrix
def generate_random_matrix(row, col, bound):
    return np.matrix(np.random.randint(-bound, bound, size=(row, col))).astype(np.int64)


def exponential(x):
    x = np.array(x)
    return 1. + x + x*x/2. + x*x*x/6.
