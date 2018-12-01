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


# Random matrix, |matrix| <= 1
def generate_random_matrix(row, col, bound):
    result = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            result[i][j] = np.random.randint(bound)
    return np.matrix(result)
