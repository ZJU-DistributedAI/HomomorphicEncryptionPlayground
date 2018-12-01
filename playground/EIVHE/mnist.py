# Import useful libraries.
from tensorflow.examples.tutorials.mnist import input_data
import random

from playground.EIVHE.layer import *
from playground.EIVHE.neural_network import *


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
    batch_size = 10
    global_sorted_indices = random.sample(range(55000), 11000)
    result = []
    settings = [(1, 0.01)]

    # Train Model 1 with the different hyper-parameter settings.
    for (num_epochs, learning_rate) in settings:

        data_set = input_data.read_data_sets("MNIST_data/", one_hot=True)  # use for training.

        eval_train_images = data_set.train.images[global_sorted_indices]
        eval_train_labels = data_set.train.labels[global_sorted_indices]
        eval_test_images = data_set.test.images
        eval_test_labels = data_set.test.labels

        #####################################################
        # Define model, loss, update and evaluation metric. #
        #####################################################
        model = [
            LinearLayer(784, 10),
            SoftmaxOutputLayer()
        ]

        # Train.
        i, train_accuracy, test_accuracy = 0, [], []
        log_period_updates = int(log_period_samples / batch_size)
        while data_set.train.epochs_completed < num_epochs:

            # Update.
            i += 1
            batch_xs, batch_ys = data_set.train.next_batch(batch_size)

            #################
            # Training step #
            #################
            perform_training(model, batch_xs, batch_ys, learning_rate)

            # Periodically evaluate.
            if i % log_period_updates == 0:
                #####################################
                # Compute and store train accuracy. #
                #####################################
                train_accuracy_value = calculate_accuracy(model, eval_train_images, eval_train_labels)
                train_accuracy.append(train_accuracy_value)
                #####################################
                # Compute and store test accuracy.  #
                #####################################
                test_accuracy_value = calculate_accuracy(model, eval_test_images, eval_test_labels)
                test_accuracy.append(test_accuracy_value)
        result.append(((num_epochs, learning_rate), train_accuracy, test_accuracy))
        print(result)
