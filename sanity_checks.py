import numpy as np
from convolution import *
from max_pooling import *


# ################################################################################
# CONVOLUTION
# ################################################################################


def test_naive_fast_convolutions():
    # Define the input of the convolution
    img = [
        [
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            [[17, 18, 19, 20], [21, 22, 23, 24], [25, 26, 27, 28], [29, 30, 31, 32]],
            [[33, 34, 35, 36], [37, 38, 39, 40], [41, 42, 43, 44], [45, 46, 47, 48]]
        ],
        [
            [[111, 211, 311, 411], [511, 611, 711, 811], [911, 1011, 1111, 121], [131, 141, 151, 161]],
            [[171, 181, 191, 201], [211, 221, 231, 241], [251, 261, 271, 281], [291, 301, 311, 321]],
            [[331, 341, 351, 361], [371, 381, 391, 401], [411, 421, 431, 441], [451, 461, 471, 481]]
        ]
    ]
    img = np.asarray(img)

    # Define the kernel
    kernel = [
        [
            # filter 1
            [[1, 2], [3, 4]],  # input channel 1
            [[5, 6], [7, 8]],  # input channel 2
            [[9, 10], [11, 12]]  # input channel 3
        ],
        [
            # filter 2
            [[13, 14], [15, 16]],  # input channel 1
            [[17, 18], [19, 20]],  # input channel 2
            [[21, 22], [23, 24]]  # input channel 3
        ]
    ]
    kernel = np.asarray(kernel)

    # Call the faster convolution version
    conv1 = fast_convolve_2d(img, kernel)

    # Call the naive convolution version
    conv2 = convolve_2d(img, kernel)

    if (conv1 == conv2).all():
        print("The 2 convolution operations gave the same result")


# ################################################################################
# MAX POOL
# ################################################################################


def max_pool_backprop_test():
    conv_data = [
        [[[3, 1, 7, 2], [5, 1, 0, 9], [8, 2, 4, 9], [4, 3, 1, 1]],
         [[3, 1, 7, 2], [9, 1, 0, 3], [5, 2, 4, 8], [4, 3, 1, 1]],
         [[3, 1, 7, 2], [5, 1, 0, 9], [8, 2, 4, 9], [4, 3, 1, 1]]]]

    conv_data = np.asarray(conv_data, dtype=np.float64)
    conv_data_shape = conv_data.shape

    maxpool_result, maxpool_pos_indices = maxPool(conv_data)
    maxpool_result_shape = maxpool_result.shape

    bp = [
        [[[1, 2], [3, 4]],
         [[5, 6], [7, 8]],
         [[9, 10], [11, 12]]]
    ]
    bp = np.asarray(bp, dtype=np.float64)
    bp_shape = bp.shape

    # 1) the gradients flowing back in the network during backprop
    #    in the network we are working with is before the flatten
    # 2) the positional indices saved during forward pass with
    #    the positions of max values
    # 3) the shape expected from the convolutional layer
    maxpool_gradients = maxpool_backprop(bp, maxpool_pos_indices, conv_data.shape)
    print()


def test_max_pool():

    data = [
        [[[3, 1, 7, 2], [5, 1, 0, 9], [8, 2, 4, 9], [4, 3, 1, 1]],
         [[3, 1, 7, 2], [9, 1, 0, 3], [5, 2, 4, 8], [4, 3, 1, 1]],
         [[3, 1, 7, 2], [5, 1, 0, 9], [8, 2, 4, 9], [4, 3, 1, 1]]],

        [[[3, 1, 7, 2], [5, 1, 0, 9], [8, 2, 4, 9], [4, 3, 1, 1]],
         [[3, 1, 7, 2], [9, 1, 0, 3], [5, 2, 4, 8], [4, 3, 1, 1]],
         [[3, 1, 7, 2], [5, 1, 0, 9], [8, 2, 4, 9], [4, 3, 1, 1]]]
    ]

    data = np.asarray(data, dtype=np.float64)

    # (2, 3, 4, 4) e.g.: 2 images, with 3 channels, 4 rows (height) and 4 columns (width)
    data_shape = data.shape

    pad = 0

    new_img, pos_result = maxPool(data, filter_h=2, filter_w=2, stride=2, padding=pad)
    # new_img = np.squeeze(new_img, axis=0)
    new_img_shape = new_img.shape
    # new_img = np.squeeze(new_img, axis=0)
    new_img = np.asarray(new_img, dtype=np.float64)
    pos_result = np.asarray(pos_result, dtype=np.int32)

    # delta_conv = np.multiply(delta_conv, dReLU(X_conv))
    print()
