# Import useful libraries.
import os

from tensorflow.examples.tutorials.mnist import input_data
import random

from EIVHE.simple_ga_nn.layer import *
from EIVHE.simple_ga_nn.neural_network import forward_step, perform_training


def calculate_accuracy(layers, images, labels):
    y_true = labels
    activations = forward_step(images, layers)
    output = np.argmax(activations[-1], axis=1)
    y_pred = np.zeros((len(y_true), 10))
    y_pred[np.arange(len(y_true)), output] = 1
    accuracy = np.sum(np.all(np.equal(y_true, y_pred), axis=1)) / float(len(y_pred))
    return np.round(accuracy, 2)


if __name__ == '__main__':
    log_period_samples = 1000
    global_sorted_indices = random.sample(range(55000), 11000)
    result = []
    settings = [(1000, 1000)]

    data_set = input_data.read_data_sets(os.getcwd() + "/MNIST_data/", one_hot=True)  # use for training.
    eval_train_images = data_set.train.images[global_sorted_indices]
    eval_train_labels = data_set.train.labels[global_sorted_indices]
    eval_test_images = data_set.test.images
    eval_test_labels = data_set.test.labels

    print('Starts training simple mnist')

    settings = {
        'batch_size': 20,
        'num_of_batches': 1000,
        'sigma':  0.01,
        'population': 40,
        'parents': 15,
        'mutation': 0.00001
    }

    # Define model, loss, update and evaluation metric.
    candidates = [[LinearLayer(784, 10), SoftmaxOutputLayer()] for _ in range(settings['population'])]
    best_model = candidates[0]

    # Train.
    i, train_accuracy, test_accuracy = 0, [], []

    best_accuracy = 0

    while i < settings['num_of_batches']:
        # Update.
        i += 1
        # Training steps
        batch_xs, batch_ys = data_set.train.next_batch(settings['batch_size'])
        candidates = perform_training(candidates, batch_xs, batch_ys, settings)
        candidates.append(best_model)

        # Evaluate
        train_accuracy_value = calculate_accuracy(candidates[0], eval_train_images, eval_train_labels)
        train_accuracy.append(train_accuracy_value)
        test_accuracy_value = calculate_accuracy(candidates[0], eval_test_images, eval_test_labels)
        test_accuracy.append(test_accuracy_value)
        if test_accuracy_value > best_accuracy:
            best_model = candidates[0]

        print(i, train_accuracy_value, test_accuracy_value)
    result.append((i, train_accuracy, test_accuracy))
