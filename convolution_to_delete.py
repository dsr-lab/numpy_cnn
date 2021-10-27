import numpy as np

from convolution import *


def col2im(gradient_values, input_shape, filter_h, filter_w, stride, pad):

    # Input size
    n_images, channels, image_h, image_w = input_shape

    # Add padding
    padded_input_h = image_h + (2 * pad)
    padded_input_w = image_w + (2 * pad)

    # Create the matrix that will contain the reshaped gradient_values
    padded_input = np.zeros((n_images, channels, padded_input_h, padded_input_w))

    # Get indices of the tensor converted into a matrix
    row_indices, col_indices, channel_matrix = get_indices(input_shape, filter_h, filter_w, stride, pad)

    a = padded_input.shape
    b = gradient_values.shape

    # Reshape the gradient in order to add the number of channels
    gradient_values_reshaped = np.array(np.hsplit(gradient_values, n_images))
    c = gradient_values_reshaped.shape

    np.add.at(padded_input, (slice(None), channel_matrix, row_indices, col_indices), gradient_values_reshaped)
    # Remove padding from new image if needed.
    if pad == 0:
        return padded_input
    elif type(pad) is int:
        return padded_input[pad:-pad, pad:-pad, :, :]

    print()

