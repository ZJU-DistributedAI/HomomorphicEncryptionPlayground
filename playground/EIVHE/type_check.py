import numpy as np


def check_is_integer(integer):
    if isinstance(integer, np.int64):
        return
    else:
        raise ValueError("must be np.int64 integer")


def check_is_vector(vector):
    if isinstance(vector, np.ndarray) and len(vector.shape) == 1:
        return
    else:
        raise ValueError("must be np.ndarray vector")


def check_is_matrix(matrix):
    if isinstance(matrix, np.matrix):
        return
    else:
        raise ValueError("must be np.matrix")


def check_is_int64(integer):
    if isinstance(integer, np.int64):
        return
    else:
        raise ValueError("must be np.int64")


def check_is_int(integer):
    if isinstance(integer, int):
        return
    else:
        raise ValueError("must be int")
