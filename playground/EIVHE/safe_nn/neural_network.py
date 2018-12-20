import random
import copy
import numpy as np


def calculate_accuracy(layers, images, labels):
    y_true = labels
    activations = simple_forward_step(images, layers)
    output = np.argmax(activations[-1], axis=1)
    y_prediction = np.zeros((len(y_true), 10))
    y_prediction[np.arange(len(y_true)), output] = 1
    accuracy = np.sum(np.all(np.equal(y_true, y_prediction), axis=1)) / float(len(y_prediction))
    return np.round(accuracy, 2)


# Forward step that never use homomorphic encryption
def simple_forward_step(input_samples, layers):
    activations = [input_samples]
    for layer in layers:
        activations.append(layer.simple_forward(activations[-1]))
    return activations


# The normal forward step that might use homomorphic encryption
def forward_step(input_samples, layers):
    activations = [input_samples]
    for layer in layers:
        activations.append(layer.forward(activations[-1]))
    return activations


def backward_step(activations, targets, layers, learning_rate):
    parameter = targets
    for index, layer in enumerate(reversed(layers)):
        y = activations.pop()
        x = activations[-1]
        parameter = layer.backward(learning_rate, y, x, parameter)


def perform_training(layers, batch_xs, batch_ys, learning_rate):
    activations = forward_step(batch_xs, layers)
    backward_step(activations, batch_ys, layers, learning_rate)


# The explore step for Genetic Algorithm
def explore(layers, settings):
    for layer in layers:
        layer.explore(settings)


# The mutate step for Genetic Algorithm
def mutate(layers, settings):
    for layer in layers:
        layer.mutate(settings)


def perform_genetic_algorithm_training(layers_candidates, batch_xs, batch_ys, settings):
    cost_list = []
    for layers in layers_candidates:
        activations = forward_step(batch_xs, layers)
        predicted_result = activations[-1]
        cost = np.sum(np.absolute(np.subtract(batch_ys, predicted_result)))
        cost_list.append(cost)
    arg_sort = np.argsort(cost_list)

    # Elitism: the best candidate will always be a parent
    new_candidates = [layers_candidates[arg_sort[i]] for i in range(settings['parents'])]

    # Subtract 1 here because the top 1 candidate is already in new candidate list
    for _ in range(settings['population'] - settings['parents']):
        parent = copy.deepcopy(random.choice(new_candidates[:settings['parents']]))
        explore(parent, settings)
        mutate(parent, settings)
        new_candidates.append(parent)
    return new_candidates
