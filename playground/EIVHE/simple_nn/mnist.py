# Import useful libraries.
import os

from tensorflow.examples.tutorials.mnist import input_data
import random
from playground.EIVHE.simple_nn.layer import *
from playground.EIVHE.simple_nn.neural_network import *


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
    batch_size = 1
    global_sorted_indices = random.sample(range(55000), 11000)
    result = []
    settings = [(40, 0.01)]

    data_set = input_data.read_data_sets(os.getcwd() + "/MNIST_data/", one_hot=True)  # use for training.
    eval_train_images = data_set.train.images[global_sorted_indices]
    eval_train_labels = data_set.train.labels[global_sorted_indices]
    eval_test_images = data_set.test.images
    eval_test_labels = data_set.test.labels

    print('Starts training simple mnist')
    for (num_of_batches, learning_rate) in settings:
        # Define model, loss, update and evaluation metric.

        # Train.
        i, train_accuracy, test_accuracy = 0, [], []

        model = [
            LinearLayer(784, 10),
            SoftmaxOutputLayer()
        ]

        while i < num_of_batches:
            # Update.
            i += 1
            # Training steps
            batch_xs, batch_ys = data_set.train.next_batch(batch_size)
            for j in range(batch_size):
                perform_training(model, batch_xs[j], batch_ys[j], learning_rate)

            # Evaluate
            train_accuracy_value = calculate_accuracy(model, eval_train_images, eval_train_labels)
            train_accuracy.append(train_accuracy_value)
            test_accuracy_value = calculate_accuracy(model, eval_test_images, eval_test_labels)
            test_accuracy.append(test_accuracy_value)

            print(i, train_accuracy_value, test_accuracy_value)
        result.append(((i, learning_rate), train_accuracy, test_accuracy))
