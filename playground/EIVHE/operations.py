from playground.EIVHE.type_check import check_is_vector


def secure_add_vectors(v1, v2):
    check_is_vector(v1)
    check_is_vector(v2)
    return v1 + v2

# def secure_linear_transfer():
#     return
#
# def secure_weighted_inner_product():
#     return