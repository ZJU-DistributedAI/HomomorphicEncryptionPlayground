import numpy as np
import copy
import random


def forward_step(input_samples, layers):
    activations = [input_samples]
    for layer in layers:
        activations.append(layer.forward(activations[-1]))
    return activations


def explore(layers, settings):
    for layer in layers:
        layer.explore(settings)


def mutate(layers, settings):
    for layer in layers:
        layer.mutate(settings)


def perform_training(layers_candidates, batch_xs, batch_ys, settings):
    cost_list = []
    new_candidates = []
    for layers in layers_candidates:
        activations = forward_step(batch_xs, layers)
        predicted_result = activations[-1]
        cost = np.sum(np.absolute(np.subtract(batch_ys, predicted_result)))
        cost_list.append(cost)
    arg_sort = np.argsort(cost_list)


    # Elitism: the best candidate will always be a parent
    # The best candidate will be the first element in the list
    parents = [layers_candidates[arg_sort[i]] for i in range(settings['parents'])]

    # Subtract 1 here because the top 1 candidate is already in new candidate list
    for _ in range(settings['population']):
        parent = copy.deepcopy(random.choice(parents))
        explore(parent, settings)
        mutate(parent, settings)
        new_candidates.append(parent)

    # print(new_candidates[0][0].b)
    return new_candidates
