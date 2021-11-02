import numpy as np
from convolution import *
from max_pooling import *
from softmax import softmax


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
    conv1 = fast_convolve_2d(img, kernel, padding=1)

    # Call the naive convolution version
    conv2 = convolve_2d(img, kernel, padding=1)

    conv1_shape = conv1.shape

    if (conv1 == conv2).all():
        print("The 2 convolution operations gave the same result")


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

    X = np.asarray(X, dtype=np.float32)
    kernel = generate_kernel(kernel_h=2, kernel_w=2, input_channels=3, output_channels=2, random=False)
    kernel2 = generate_kernel(kernel_h=2, kernel_w=2, input_channels=1, output_channels=2, random=False)

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
    gradient_values = np.asarray(gradient_values, dtype=np.float32)

    dw1, dx1 = fast_convolution_backprop(X, kernel, gradient_values)
    dw2, dx2 = convolution_backprop(X, kernel, gradient_values)

    if (convolution_result == convolution_result2).all():
        print("CONV EQUAL")
    else:
        print("CONV NOT EQUAL")

    if (dw1 == dw2).all() and (dx1 == dx2).all():
        print("BACKPROB EQUAL")
    else:
        print("BACKPROP NOT EQUAL")



    print()


# ################################################################################
# MAX POOL
# ################################################################################


def max_pool_backprop_test():
    # The output of the conv layer
    conv_data = [
        [[[3, 1, 7, 2], [5, 1, 0, 9], [8, 2, 4, 9], [4, 3, 1, 1]],
         [[12, 18, 0, 2], [33, 2, 55, 60], [5, 2, 4, 8], [4, 32, 101, 1]],
         [[19, 1, 27, 2], [51, 1, 0, 9], [82, 2, 4, 9], [4, 3, 1, 11]]],
        #[[[3, 1, 7, 2], [5, 1, 0, 9], [8, 2, 4, 9], [4, 3, 1, 1]],
        #[[12, 18, 0, 2], [33, 2, 55, 60], [5, 2, 4, 8], [4, 32, 101, 1]],
         #[[19, 1, 27, 2], [51, 1, 0, 9], [82, 2, 4, 9], [4, 3, 1, 11]]]
    ]

    conv_data = np.asarray(conv_data, dtype=np.float32)
    conv_data_shape = conv_data.shape

    # maxpool_result, maxpool_pos_indices = max_pool(conv_data)
    # maxpool_result2, maxpool_pos_indices2 = fast_maxpool_backprop(conv_data,)

    pad = 0
    new_img, pos_idx = max_pool(conv_data, filter_h=2, filter_w=2, stride=2, padding=pad)
    new_img2, pos_idx2 = fast_max_pool(conv_data, kernel_h=2, kernel_w=2, stride=2, padding=pad)

    if (new_img == new_img2).all():
        print("The 2 max pooling operations gave the same result")

    bp = [
        [[[1, 2], [3, 4]],
         [[5, 6], [7, 8]],
         [[9, 10], [11, 12]]],
        #[[[1, 2], [3, 4]],
         #[[5, 6], [7, 8]],
         #[[9, 10], [11, 12]]]
    ]
    bp = np.asarray(bp, dtype=np.float32)
    bp_shape = bp.shape

    # 1) the gradients flowing back in the network during backprop
    #    in the network we are working with is before the flatten
    # 2) the positional indices saved during forward pass with
    #    the positions of max values
    # 3) the shape expected from the convolutional layer
    maxpool_gradients = maxpool_backprop(bp, pos_idx, conv_data.shape)
    maxpool_gradients2 = fast_maxpool_backprop(bp, conv_data.shape, padding=0, stride=2, max_pool_size=2, pos_result=pos_idx2)

    if(maxpool_gradients == maxpool_gradients2).all():
        print("backprop same result!!!!!")

    print()

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

    data = np.asarray(data, dtype=np.float32)

    # (2, 3, 4, 4) e.g.: 2 images, with 3 channels, 4 rows (height) and 4 columns (width)
    data_shape = data.shape

    pad = 0

    new_img, pos_result = max_pool(data, filter_h=2, filter_w=2, stride=2, padding=pad)
    # new_img = np.squeeze(new_img, axis=0)
    new_img_shape = new_img.shape
    # new_img = np.squeeze(new_img, axis=0)
    new_img = np.asarray(new_img, dtype=np.float32)
    pos_result = np.asarray(pos_result, dtype=np.int32)

    # delta_conv = np.multiply(delta_conv, dReLU(X_conv))
    print()


def test_naive_fast_max_pool():
    data = [
        [[[3, 1, 7, 2], [5, 1, 0, 9], [8, 2, 4, 9], [4, 3, 1, 1]],
         [[3, 1, 7, 2], [9, 1, 0, 3], [5, 2, 4, 8], [4, 3, 1, 1]],
         [[3, 1, 7, 2], [5, 1, 0, 9], [8, 2, 4, 9], [4, 3, 1, 1]]],

        # [[[3, 1, 7, 2], [5, 1, 0, 9], [8, 2, 4, 9], [4, 3, 1, 1]],
        #  [[3, 1, 7, 2], [9, 1, 0, 3], [5, 2, 4, 8], [4, 3, 1, 1]],
        #  [[3, 1, 7, 2], [5, 1, 0, 9], [8, 2, 4, 9], [4, 3, 1, 1]]]
    ]
    data = [  # number of images
        [  # number of channels
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],  # heigth and width of the input
            [[21, 22, 23, 24], [25, 26, 27, 28], [29, 30, 31, 32], [33, 34, 35, 36]],
            [[41, 42, 43, 44], [45, 46, 47, 48], [49, 50, 51, 52], [53, 54, 55, 56]],
        ]
    ]
    data = np.asarray(data, dtype=np.float32)
    a = data.shape
    data = np.random.randint(0, high=255, size=(1, 3, 4, 4))
    b = data.shape


    # (2, 3, 4, 4) e.g.: 2 images, with 3 channels, 4 rows (height) and 4 columns (width)
    data_shape = data.shape

    pad = 0

    new_img, pos_idx = fast_max_pool(data, kernel_h=2, kernel_w=2, stride=2, padding=pad)
    new_img2, pos_idx = max_pool(data, filter_h=2, filter_w=2, stride=2, padding=pad)

    if (new_img == new_img2).all():
        print("The 2 max pooling operations gave the same result")

    print()


# ################################################################################
# SOFTMAX
# ################################################################################
def test_softmax():
    scores = [[3, -2, -100], [3, -2, -100]]
    scores = [[1, 2, 3, 6],
              [2, 4, 5, 6],
              [3, 8, 7, 6]]
    scores = [[-1, 0, 3, 5]]

    result = softmax(scores)
    print(result)
    print('softmax computed')

# ################################################################################
# GRADIENT CHECK
# ################################################################################

def gradient_check(x, theta, J_plus, J_minus, grad, epsilon=1e-7):


    #J_plus = forward_propagation(x, thetaplus)
    #J_minus = forward_propagation(x, thetaminus)

    gradapprox = (J_plus - J_minus) / (2 * epsilon)

    # Check if gradapprox is close enough to backward propagation
    #grad = backward_propagation(x, theta)

    numerator = np.linalg.norm(grad - gradapprox)
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
    difference = numerator / denominator

    if difference < 1e-7:
        print('The gradient is correct')
    else:
        print('The gradient is wrong')

    return difference