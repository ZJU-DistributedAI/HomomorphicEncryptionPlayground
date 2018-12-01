import time

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from playground.EIVHE.eivhe_helper import EIVHE, generate_random_matrix

if __name__ == '__main__':
    data_set = input_data.read_data_sets("MNIST_data/", one_hot=True)
    eval_train_images = data_set.train.images
    eval_train_labels = data_set.train.labels
    eivhe = EIVHE(7, np.int64(10), np.int64(10))
    # Pick x and w
    x = (eval_train_images[0] * 100).round().astype('int64')[0:728]
    w = np.int64(10000)
    # Naively create s and c
    s0 = eivhe.naive_encrypt_secret(w, x.size)
    c0 = x
    # Key switching, s0c0 = s1c1
    t1 = generate_random_matrix(728, 1, 10)
    s1 = eivhe.key_switching_get_secret(t1)
    m1 = eivhe.key_switching_get_switching_matrix(s0, t1)
    start_time = time.time()
    c1 = eivhe.key_switching_get_cipher_from_switching_matrix(c0, m1)
    np.testing.assert_equal(x, eivhe.decrypt(s1, c1, w))