# Import useful libraries.
import os

from tensorflow.examples.tutorials.mnist import input_data
import copy

from EIVHE.encryption import Encryption
from EIVHE.encryption_core import EncryptionCore
from EIVHE.safe_nn.layer import *
from EIVHE.safe_nn.neural_network import *
from EIVHE.safe_nn.safe_layer import *
from EIVHE.safe_nn.simple_layer import *

if __name__ == '__main__':
    log_period_samples = 1000
    global_sorted_indices = random.sample(range(55000), 11000)
    result = []
    settings = {
        'batch_size': 1,
        'num_of_batches': 1000,
        'training_method': 'simple',
        'simple_training_params': {
            'learning_rate': 0.01
        }
    }

    # settings = [(1000, 0.01, 'genetic_algorithm')]

    data_set = input_data.read_data_sets(os.getcwd() + "/MNIST_data/", one_hot=True)  # use for training.
    eval_train_images = data_set.train.images[global_sorted_indices]
    eval_train_labels = data_set.train.labels[global_sorted_indices]
    eval_test_images = data_set.test.images
    eval_test_labels = data_set.test.labels

    print('Starts training secure mnist')

    # Init encryption instance
    number_of_bits = 40
    a_bound = np.int64(5)
    e_bound = np.int64(5)
    encryption_core = EncryptionCore(number_of_bits, a_bound, e_bound)
    t_bound = np.int64(10)
    scale = 100
    w = np.int64(2 ** 10)
    input_range = 1000
    encryption = Encryption(encryption_core, w, scale, t_bound, input_range)
    # Finished configuring encryption instance



    # Train.
    i, train_accuracy, test_accuracy = 0, [], []

    model = [
        LinearLayer(784, 50),
        ReluLayer(),
        LinearLayer(50, 50),
        ReluLayer(),
        # LinearLayer(50, 10),
        # SoftmaxOutputLayer()

        HomomorphicEncryptionLayer(encryption),
        SafeLinearLayer(encryption, 50, 10),
        SafeSoftmaxOutputLayer(encryption),
        HomomorphicDecryptionLayer(encryption)

    ]

    if settings['training_method'] == 'genetic_algorithm':
        candidates = [copy.deepcopy(model) for _ in range(settings['population'])]

    while i < settings['num_of_batches']:
        # Update.
        i += 1
        # Training steps
        batch_xs, batch_ys = data_set.train.next_batch(settings['batch_size'])
        for j in range(settings['batch_size']):
            if settings['training_method'] == 'simple':
                perform_training(model, batch_xs[j], batch_ys[j], settings['simple_training_params']['learning_rate'])
            if settings['training_method'] == 'genetic_algorithm':
                perform_genetic_algorithm_training()

        # Evaluate
        train_accuracy_value = calculate_accuracy(model, eval_train_images, eval_train_labels)
        train_accuracy.append(train_accuracy_value)
        test_accuracy_value = calculate_accuracy(model, eval_test_images, eval_test_labels)
        test_accuracy.append(test_accuracy_value)

        print(i, train_accuracy_value, test_accuracy_value)
    result.append(i, train_accuracy, test_accuracy)
