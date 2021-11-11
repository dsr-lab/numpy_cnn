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
        # [
        #    [[111, 211, 311, 411], [511, 611, 711, 811], [911, 1011, 1111, 121], [131, 141, 151, 161]],
        #    #[[171, 181, 191, 201], [211, 221, 231, 241], [251, 261, 271, 281], [291, 301, 311, 321]],
        #    #[[331, 341, 351, 361], [371, 381, 391, 401], [411, 421, 431, 441], [451, 461, 471, 481]]
        # ]
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

    # def generate_kernel(input_channels=3, output_channels=16, kernel_h=3, kernel_w=3, random=True):
    # kernel = np.ones((1, 1, 2, 2))

    padding = 0

    # Call the faster convolution version
    conv1 = fast_convolve_2d(img, kernel, padding=padding)

    # Call the naive convolution version
    conv2 = convolve_2d(img, kernel, padding=padding)

    conv1_shape = conv1.shape

    if (conv1 == conv2).all():
        print("The 2 convolution operations gave the same result")


def relu_back(x, dout):
    dx = np.array(dout, copy=True)
    dx[x <= 0] = 0

    return dx


def convolution_method_comparisons(X, kernel, gradient_values):
    # 2 images, 3 channels and 4x4 size image
    # NOTE: Kernel MUST have 3 input channels
    # X = [
    #     [
    #         [[3, 1, 7, 2], [5, 1, 0, 9], [8, 2, 4, 9], [4, 3, 1, 1]],
    #         [[3, 1, 7, 2], [9, 1, 0, 3], [5, 2, 4, 8], [4, 3, 1, 1]],
    #         [[3, 1, 7, 2], [5, 1, 0, 9], [8, 2, 4, 9], [4, 3, 1, 1]]
    #     ],
    #     [
    #         [[3, 1, 7, 2], [5, 1, 0, 9], [8, 2, 4, 9], [4, 3, 1, 1]],
    #         [[3, 1, 7, 2], [9, 1, 0, 3], [5, 2, 4, 8], [4, 3, 1, 1]],
    #         [[3, 1, 7, 2], [5, 1, 0, 9], [8, 2, 4, 9], [4, 3, 1, 1]]
    #     ]
    # ]

    # 1 image, 1 channel and 4x4 size image
    # NOTE: Kernel MUST have 1 input channel
    # X = [
    #     [[[3, 1, 7, 2], [5, 1, 0, 9], [8, 2, 4, 9], [4, 3, 1, 1]]]
    # ]

    # X = np.asarray(X, dtype=np.float32)
    # #kernel = generate_kernel(kernel_h=2, kernel_w=2, input_channels=3, output_channels=2, random=False)
    #
    # kernel = [
    #     [
    #         # filter 1
    #         [[1, 2], [3, 4]],  # input channel 1
    #         [[5, 6], [7, 8]],  # input channel 2
    #         [[9, 10], [11, 12]]  # input channel 3
    #     ],
    #     [
    #         # filter 2
    #         [[13, 14], [15, 16]],  # input channel 1
    #         [[17, 18], [19, 20]],  # input channel 2
    #         [[21, 22], [23, 24]]  # input channel 3
    #     ]
    # ]
    # kernel = np.asarray(kernel)

    convolution_result = convolve_2d(X, kernel)
    convolution_result2 = fast_convolve_2d(X, kernel)
    convolution_result3 = conv_forward_naive(X, kernel)

    a = np.isclose(convolution_result, convolution_result2).all()
    b = np.isclose(convolution_result, convolution_result3).all()
    print()

    # Valid gradient size for: 2 images, 2 channels output in the kernel
    # gradient_values = [
    #     [
    #         [[3, 1, 4], [5, 1, 7], [5, 1, 7]],
    #         [[1, 2, 3], [1, 0, 2], [9, 1, 1]]
    #     ],
    #     [
    #         [[3, 1, 4], [5, 1, 7], [5, 1, 7]],
    #         [[1, 2, 3], [1, 0, 2], [9, 1, 1]]
    #     ]
    # ]
    # Valid gradient size for: 1 image, 1 channel output in the kernel
    # gradient_values = [
    #     [
    #         [[3, 1, 4], [5, 1, 7], [5, 1, 7]],
    #         [[1, 2, 3], [1, 0, 2], [9, 1, 1]]
    #     ]
    # ]
    # gradient_values = np.asarray(gradient_values, dtype=np.float32)

    dw1, dx1 = fast_convolution_backprop(X, kernel, gradient_values)
    dw2, dx2 = convolution_backprop(X, kernel, gradient_values)
    cache = (X, kernel, 0, 1)
    dw3, dx3 = conv_back_naive(gradient_values, cache)

    if (convolution_result == convolution_result2).all() \
            and (convolution_result == convolution_result3).all():
        print("CONV EQUAL")
    else:
        print("CONV NOT EQUAL")

    if (dw1 == dw2).all() and (dx1 == dx2).all():
        print("BACKPROB EQUAL")
    else:
        print("BACKPROP NOT EQUAL")

    if (dw1 == dw3).all() and (dx1 == dx3).all():
        print("BACKPROB EQUAL")
    else:
        print("BACKPROP NOT EQUAL")
    aa = np.isclose(dw1, dw2).all()
    bb = np.isclose(dw1, dw3).all()
    cc = np.isclose(dx1, dx2).all()
    dd = np.isclose(dx1, dx3).all()

    print()


def conv_forward_naive(x, weight):
    pad = 0
    stride = 1

    (m, n_C_prev, n_h, n_w) = x.shape
    (n_C, n_C_prev, f, f) = weight.shape

    n_H = int(1 + (n_h + 2 * pad - f) / stride)
    n_W = int(1 + (n_w + 2 * pad - f) / stride)

    x_prev_pad = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)

    Z = np.zeros((m, n_C, n_H, n_W))

    caches = (x, weight, pad, stride)

    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    x_slice = x_prev_pad[i, :, vert_start:vert_end, horiz_start:horiz_end]
                    Z[i, c, h, w] = np.sum(np.multiply(x_slice, weight[c, :, :, :]))

    return Z


def conv_back_naive(dout, cache):
    x, w_filter, pad, stride = cache

    (m, n_C_prev, n_h, n_w) = x.shape
    (n_C, n_C_prev, f, f) = w_filter.shape

    n_H = int(1 + (n_h + 2 * pad - f) / stride)
    n_W = int(1 + (n_w + 2 * pad - f) / stride)

    a_prev_pad = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)

    dw = np.zeros(w_filter.shape, dtype=np.float)
    dx = np.zeros(x.shape, dtype=np.float)

    for h in range(f):
        for w in range(f):
            for p in range(n_C_prev):
                for c in range(n_C):
                    # go through all the individual positions that this filter affected and multiply by their dout
                    a_slice = a_prev_pad[:, p, h:h + n_H * stride:stride, w:w + n_W * stride:stride]

                    dw[c, p, h, w] = np.sum(a_slice * dout[:, c, :, :])

    # TODO: put back in dout to get correct gradient
    dx_pad = np.pad(dx, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)

    for i in range(m):
        for h_output in range(n_H):
            for w_output in range(n_W):
                for g in range(n_C):
                    vert_start = h_output * stride
                    vert_end = vert_start + f
                    horiz_start = w_output * stride
                    horiz_end = horiz_start + f
                    dx_pad[i, :, vert_start:vert_end, horiz_start:horiz_end] += w_filter[g, :, :, :] * dout[
                        i, g, h_output, w_output]

    dx = dx_pad[:, pad:pad + n_h, pad:pad + n_w, :]

    # db = np.sum(dout, axis=(0, 1, 2))

    return dw, dx  # , db


# ################################################################################
# MAX POOL
# ################################################################################

def max_pooling_back(dout, caches):
    pool, prev, filter_size = caches

    (m, channels, n_H, n_W) = pool.shape
    (m_prev, channels_prev, n_prev_H, n_prev_W) = prev.shape

    empty = np.zeros((m, channels, n_prev_H, n_prev_W))

    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(channels):
                    vert_start = h * filter_size
                    vert_end = vert_start + filter_size
                    horiz_start = w * filter_size
                    horiz_end = horiz_start + filter_size

                    mask = prev[i, c, vert_start:vert_end, horiz_start:horiz_end] == pool[i, c, h, w]

                    empty[i, c, vert_start:vert_end, horiz_start:horiz_end] = mask * dout[i, c, h, w]

    return empty


def max_pool_backprop_test():
    # The output of the conv layer
    conv_data = [
        [[[3, 1, 7, 2], [5, 1, 0, 9], [8, 2, 4, 9], [4, 3, 1, 1]],
         [[12, 18, 0, 2], [33, 2, 55, 60], [5, 2, 4, 8], [4, 32, 101, 1]],
         [[19, 1, 27, 2], [51, 1, 0, 9], [82, 2, 4, 9], [4, 3, 1, 11]]],
        [[[3, 1, 7, 2], [5, 1, 0, 9], [8, 2, 4, 9], [4, 3, 1, 1]],
         [[12, 18, 0, 2], [33, 2, 55, 60], [5, 2, 4, 8], [4, 32, 101, 1]],
         [[19, 1, 27, 2], [51, 1, 0, 9], [82, 2, 4, 9], [4, 3, 1, 11]]]
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
        # [[[1, 2], [3, 4]],
        # [[5, 6], [7, 8]],
        # [[9, 10], [11, 12]]]
    ]
    bp = np.asarray(bp, dtype=np.float32)
    bp_shape = bp.shape

    # 1) the gradients flowing back in the network during backprop
    #    in the network we are working with is before the flatten
    # 2) the positional indices saved during forward pass with
    #    the positions of max values
    # 3) the shape expected from the convolutional layer
    maxpool_gradients = maxpool_backprop(bp, pos_idx, conv_data.shape)
    maxpool_gradients2 = fast_maxpool_backprop(bp, conv_data.shape, padding=0, stride=2, max_pool_size=2,
                                               pos_result=pos_idx2)

    if (maxpool_gradients == maxpool_gradients2).all():
        print("backprop same result!!!!!")

    print()

    print()


def max_pooling_n(prev_layer, filter_size=2, stride=2, padding=0):
    (m, channels, n_H_prev, n_W_prev) = prev_layer.shape

    # with max pooling I dont want overlapping filters so make stride = filter size
    n_H = int((n_H_prev - filter_size) / filter_size + 1)
    n_W = int((n_W_prev - filter_size) / filter_size + 1)

    pooling = np.zeros((m, channels, n_H, n_W))

    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(channels):
                    vert_start = h * filter_size
                    vert_end = vert_start + filter_size
                    horiz_start = w * filter_size
                    horiz_end = horiz_start + filter_size

                    prev_slice = prev_layer[i, c, vert_start:vert_end, horiz_start:horiz_end]

                    pooling[i, c, h, w] = np.max(prev_slice)

    caches = (pooling, prev_layer, filter_size)

    return pooling, caches


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


def test_naive_fast_max_pool(data):
    # data = [
    #     [[[3, 1, 7, 2], [5, 1, 0, 9], [8, 2, 4, 9], [4, 3, 1, 1]],
    #      [[3, 1, 7, 2], [9, 1, 0, 3], [5, 2, 4, 8], [4, 3, 1, 1]],
    #      [[3, 1, 7, 2], [5, 1, 0, 9], [8, 2, 4, 9], [4, 3, 1, 1]]],
    #
    #     # [[[3, 1, 7, 2], [5, 1, 0, 9], [8, 2, 4, 9], [4, 3, 1, 1]],
    #     #  [[3, 1, 7, 2], [9, 1, 0, 3], [5, 2, 4, 8], [4, 3, 1, 1]],
    #     #  [[3, 1, 7, 2], [5, 1, 0, 9], [8, 2, 4, 9], [4, 3, 1, 1]]]
    # ]
    # data = [  # number of images
    #     [  # number of channels
    #         [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],  # heigth and width of the input
    #         [[21, 22, 23, 24], [25, 26, 27, 28], [29, 30, 31, 32], [33, 34, 35, 36]],
    #         [[41, 42, 43, 44], [45, 46, 47, 48], [49, 50, 51, 52], [53, 54, 55, 56]],
    #     ]
    # ]
    # data = np.asarray(data, dtype=np.float32)
    # a = data.shape
    # data = np.random.randint(0, high=255, size=(1, 3, 4, 4))
    # b = data.shape

    # (2, 3, 4, 4) e.g.: 2 images, with 3 channels, 4 rows (height) and 4 columns (width)
    data_shape = data.shape

    pad = 0

    new_img, pos_idx = max_pool(data, filter_h=2, filter_w=2, stride=2, padding=pad)
    new_img2, pos_idx2 = fast_max_pool(data, kernel_h=2, kernel_w=2, stride=2, padding=pad)
    new_img3, caches = max_pooling_n(data, filter_size=2, stride=2, padding=pad)

    if (new_img == new_img2).all() and (new_img == new_img3).all():
        print("The 2 max pooling operations gave the same result")
    else:
        print("Different maxpool")

    np.random.seed(100)
    gradient_values = np.random.randint(0, 50,
                                        (new_img.shape[0], new_img.shape[1], new_img.shape[2], new_img.shape[3]))

    max_pool_back1 = maxpool_backprop(gradient_values, pos_idx, data.shape)
    max_pool_back2 = fast_maxpool_backprop(gradient_values, data.shape, padding=0, stride=2, max_pool_size=2,
                                           pos_result=pos_idx2)
    max_pool_back3 = max_pooling_back(gradient_values, caches)

    if (max_pool_back1 == max_pool_back2).all():
        print("EQUAL!")

    if (max_pool_back1 == max_pool_back3).all():
        print("EQUAL!")

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
    # J_plus = forward_propagation(x, thetaplus)
    # J_minus = forward_propagation(x, thetaminus)

    gradapprox = (J_plus - J_minus) / (2 * epsilon)

    # Check if gradapprox is close enough to backward propagation
    # grad = backward_propagation(x, theta)

    numerator = np.linalg.norm(grad - gradapprox)
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
    difference = numerator / denominator

    if difference < 1e-7:
        print('The gradient is correct')
    else:
        print('The gradient is wrong')

    return difference
