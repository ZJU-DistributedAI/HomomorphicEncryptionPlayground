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


def perform_simple_training(layers, batch_xs, batch_ys, settings):
    learning_rate = settings['simple_training_params']['learning_rate']
    activations = forward_step(batch_xs, layers)
    backward_step(activations, batch_ys, layers, learning_rate)


# The explore step for Genetic Algorithm
def explore(layers, sigma):
    for layer in layers:
        layer.explore(sigma)


# The mutate step for Genetic Algorithm
def mutate(layers, mutation_probability):
    for layer in layers:
        layer.mutate(mutation_probability)


def perform_genetic_algorithm_training(candidates, batch_xs, batch_ys, settings):
    population_size = settings['genetic_algorithm_params']['population']
    parents_size = settings['genetic_algorithm_params']['parents']
    sigma = settings['genetic_algorithm_params']['sigma']
    mutation_probability = settings['genetic_algorithm_params']['mutation_probability']
    cost_list = []

    for model in candidates:
        activations = forward_step(batch_xs, model)
        predicted_result = activations[-1]
        cost = np.sum(np.absolute(np.subtract(batch_ys, predicted_result)))
        cost_list.append(cost)
    arg_sort = np.argsort(cost_list)

    # Elitism: the best candidate will always be a parent
    # Add all parents to candidates, sorted by performance
    new_candidates = [candidates[arg_sort[i]] for i in range(parents_size)]

    # Fill the remaining slots by (randomly generated) new candidates
    for _ in range(population_size - parents_size):
        parent = copy.deepcopy(random.choice(new_candidates[:parents_size]))
        explore(parent, sigma)
        mutate(parent, mutation_probability)
        new_candidates.append(parent)
    return new_candidates
