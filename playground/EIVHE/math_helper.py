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
# note that this vector is vertical
def matrix_to_vector(matrix):
    check_is_matrix(matrix)
    n_rows = matrix.shape[0]
    n_cols = matrix.shape[1]
    result = np.zeros(matrix.size)
    for i in range(n_rows):
        for j in range(n_cols):
            result[i * n_rows + j] = matrix[i, j]
    return result


# Random matrix, |matrix| <= 1
def generate_random_matrix(row, col, bound):
    result = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            result[i][j] = np.random.randint(bound)
    return np.matrix(result)
