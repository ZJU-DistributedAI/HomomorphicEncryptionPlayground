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





