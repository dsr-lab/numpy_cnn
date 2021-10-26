import numpy as np

from convolution import *


def convolution_method_comparisons():

    # 2 images, 3 channels and 4x4 size image
    # NOTE: Kernel MUST have 3 input channels
    X = [
        [
            [[3, 1, 7, 2], [5, 1, 0, 9], [8, 2, 4, 9], [4, 3, 1, 1]],
            [[3, 1, 7, 2], [9, 1, 0, 3], [5, 2, 4, 8], [4, 3, 1, 1]],
            [[3, 1, 7, 2], [5, 1, 0, 9], [8, 2, 4, 9], [4, 3, 1, 1]]
        ],
        [
            [[3, 1, 7, 2], [5, 1, 0, 9], [8, 2, 4, 9], [4, 3, 1, 1]],
            [[3, 1, 7, 2], [9, 1, 0, 3], [5, 2, 4, 8], [4, 3, 1, 1]],
            [[3, 1, 7, 2], [5, 1, 0, 9], [8, 2, 4, 9], [4, 3, 1, 1]]
        ]
    ]

    # 1 image, 1 channel and 4x4 size image
    # NOTE: Kernel MUST have 1 input channel
    # X = [
    #     [[[3, 1, 7, 2], [5, 1, 0, 9], [8, 2, 4, 9], [4, 3, 1, 1]]]
    # ]

    X = np.asarray(X, dtype=np.float64)
    kernel = init_random_kernel(kernel_h=2, kernel_w=2, input_channels=3, random=False)
    kernel2 = init_random_kernel(kernel_h=2, kernel_w=2, input_channels=1, random=False)

    convolution_result = convolve_2d(X, kernel)
    convolution_result2 = convolve_2d(X, kernel2)

    # Valid gradient size for: 2 images, 2 channels output in the kernel
    gradient_values = [
        [
            [[3, 1, 4], [5, 1, 7], [5, 1, 7]],
            [[1, 2, 3], [1, 0, 2], [9, 1, 1]]
        ],
        [
            [[3, 1, 4], [5, 1, 7], [5, 1, 7]],
            [[1, 2, 3], [1, 0, 2], [9, 1, 1]]
        ]
    ]
    # Valid gradient size for: 1 image, 1 channel output in the kernel
    # gradient_values = [
    #     [
    #         [[3, 1, 4], [5, 1, 7], [5, 1, 7]],
    #         [[1, 2, 3], [1, 0, 2], [9, 1, 1]]
    #     ]
    # ]
    gradient_values = np.asarray(gradient_values, dtype=np.float64)

    a = test_conv_back(X, kernel, gradient_values)
    b = convolution_backprop(X, kernel, gradient_values)

    if (convolution_result == convolution_result2).all():
        print("CONV EQUAL")
    else:
        print("CONV NOT EQUAL")

    if (a == b).all():
        print("BACKPROB EQUAL")
    else:
        print("BACKPROP NOT EQUAL")

    print()


def test_conv_back(X_batch, kernel, gradient_values):

    X_batch_im2col = im2col(X=X_batch, conv1=kernel, stride=1, pad=0)
    delta_conv_reshape = error_layer_reshape(gradient_values)
    conv1_delta = (delta_conv_reshape @ X_batch_im2col.T).reshape(kernel.shape)

    return conv1_delta


def error_layer_reshape(error_layer):
    test_array = error_layer
    test_array_new = np.zeros((test_array.shape[1], test_array.shape[0] * test_array.shape[2] * test_array.shape[3]))
    for i in range(test_array_new.shape[0]):
        test_array_new[i:i + 1, :] = test_array[:, i:i + 1, :, :].ravel()
    return test_array_new


def get_indices(X_shape, filter_h, filter_w, stride, pad):

    # Input size
    n_images, channels, image_h, image_w = X_shape

    # Output size
    out_h = int((image_h + 2 * pad - filter_h) / stride) + 1
    out_w = int((image_w + 2 * pad - filter_w) / stride) + 1

    # ##############################
    # Row indices
    # ##############################

    # Create the starting point for rows indices
    # The goal is to create an array that has:
    # - values that go from 0 to (filter_h - 1)
    # - repeat the above values (filter_w - 1) times
    #
    # Example 1:
    #   filter_h = filter_w = 2
    #   a = np.arange(filter_h) = [0, 1]
    #   np.repeat(a) = [0, 0, 1, 1]
    #
    # Example 2:
    #   filter_h = 3, filter_w = 2
    #   a = np.arange(filter_h) = [0, 1, 2]
    #   np.repeat(a) = [0, 0, 1, 1, 2, 2]
    row_indices_vector_0 = np.repeat(np.arange(filter_h), filter_w)

    # Repeat based on the number of channels
    # Example:
    #   filter_h = filter_w = 2, channels = 3
    #   a = np.arange(filter_h) = [0, 1]
    #   b = np.repeat(a) = [0, 0, 1, 1]
    #   np.tile(b, channels) = [[0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]

    row_indices_vector_0 = np.tile(row_indices_vector_0, channels)

    # Create the vector that is used for summing 1 after each level of the
    # convolution operation
    sum_vector = stride * np.repeat(np.arange(out_h), out_w)

    # At this point we need to sum rows_indices_vector_0 with the sum_vector.
    # Notice that:
    # - rows_indices_vector_0 is reshaped to a single column
    # - sum_vector is reshaped to a single row
    row_indices = row_indices_vector_0.reshape(-1, 1) + sum_vector.reshape(1, -1)

    # ##############################
    # Column indices
    # ##############################

    # As before, create the initial vector required for column indices
    # Differently from before, when we slide horizontally the filter we have
    # to increase the index a number of times equal to (filter_h-1)
    column_indices_vector_0 = np.tile(np.arange(filter_h), filter_w)
    column_indices_vector_0 = np.tile(column_indices_vector_0, channels)

    # Create the sum vector
    sum_vector = stride * np.tile(np.arange(out_h), out_w)

    # Sum the two vectors
    column_indices = column_indices_vector_0.reshape(-1, 1) + sum_vector.reshape(1, -1)

    # ----Compute matrix of index d----

    # Matrix required for considering different channels while considering
    # the row_indices and column_indices variables
    channel_matrix = np.repeat(np.arange(channels), filter_h * filter_w).reshape(-1, 1)

    return row_indices, column_indices, channel_matrix


def im2col_(images, filter_h, filter_w, stride, pad):

    # Apply the padding
    padded_images = np.pad(images, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    row_indices, col_indices, channel_matrix = get_indices(images.shape, filter_h, filter_w, stride, pad)

    # Apply the indexing to all the images, creating a new matrix for each image
    image_matrices = padded_images[:, channel_matrix, row_indices, col_indices]

    # Create a single matrix that considers all the images concatenating along the last axis
    image_matrices = np.concatenate(image_matrices, axis=-1)

    return image_matrices
