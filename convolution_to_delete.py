import numpy as np

from convolution import *


def convolution_comparisons():

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
    b = backprop_test(X, kernel, gradient_values)

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


def im2col(X, conv1, stride, pad):
    """
        Transforms our input image into a matrix.

        Parameters:
        - X: input image.
        - HF: filter height.
        - WF: filter width.
        - stride: stride value.
        - pad: padding value.

        Returns:
        -cols: output matrix.
    """
    # Padding
    X_padded = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    X = X_padded
    new_height = int((X.shape[2] + (2 * pad) - (conv1.shape[2])) / stride) + 1
    new_width = int((X.shape[3] + (2 * pad) - (conv1.shape[3])) / stride) + 1
    im2col_vector = np.zeros((X.shape[1] * conv1.shape[2] * conv1.shape[3], new_width * new_height * X.shape[0]))
    c = 0
    for position in range(X.shape[0]):

        image_position = X[position, :, :, :]
        for height in range(0, image_position.shape[1], stride):
            image_rectangle = image_position[:, height:height + conv1.shape[2], :]
            if image_rectangle.shape[1] < conv1.shape[2]:
                continue
            else:
                for width in range(0, image_rectangle.shape[2], stride):
                    image_square = image_rectangle[:, :, width:width + conv1.shape[3]]
                    if image_square.shape[2] < conv1.shape[3]:
                        continue
                    else:
                        im2col_vector[:, c:c + 1] = image_square.reshape(-1, 1)
                        c = c + 1

    return im2col_vector
