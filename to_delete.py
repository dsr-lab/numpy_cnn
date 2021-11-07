import numpy as np

from convolution import fast_convolve_2d, accuracy, generate_kernel, fast_convolution_backprop
from cross_entropy import cross_entropy
from flatten import flatten
from max_pooling import fast_max_pool, fast_maxpool_backprop
from relu import ReLU, dReLU
from softmax import softmax

learning_rate = 1e-3

eps = 1e-8
beta1 = 0.9
beta2 = 0.999
momentum_w1 = 0
momentum_w2 = 0
momentum_b0 = 0
momentum_b1 = 0
momentum_conv1 = 0
momentum_conv2 = 0
velocity_w1 = 0
velocity_w2 = 0
velocity_b0 = 0
velocity_b1 = 0
velocity_conv1 = 0
velocity_conv2 = 0
t = 1


def forward(input_data, input_labels, kernel, kernel2, fc1_w, fc1_b, fc2_w, fc2_b, one_hot_encoding_labels):
    # ################################################################################
    # FORWARD PASS
    # ################################################################################

    # ********************
    # CONV 1 + RELU
    # ********************

    x_conv = fast_convolve_2d(input_data, kernel, padding=0)
    conv2_input = ReLU(x_conv)

    # ********************
    # CONV 2 + RELU
    # ********************
    x_conv2 = fast_convolve_2d(conv2_input, kernel2, padding=0)
    conv_out_shape2 = x_conv2.shape

    maxpool_input = ReLU(x_conv2)

    # ********************
    # MAXPOOL
    # ********************
    x_maxpool, pos_maxpool_pos = fast_max_pool(maxpool_input)

    # ********************
    # FLATTEN + FCs
    # ********************
    fc1_input = flatten(x_maxpool)

    # First fc layer
    fc1 = np.matmul(fc1_w, fc1_input) + fc1_b
    fc2_input = ReLU(fc1)

    # Second fc layer
    fc2 = np.matmul(fc2_w, fc2_input) + fc2_b

    # Apply the softmax for computing the scores
    scores = softmax(fc2)

    # Compute the cross entropy loss
    ce = cross_entropy(scores, one_hot_encoding_labels, input_labels) * input_data.shape[0]

    # Compute prediction and accuracy
    acc = accuracy(scores, input_labels) * input_data.shape[0]

    cost = np.mean(softmax_cost(one_hot_encoding_labels, scores))

    return scores, fc2_input, fc1, fc1_input, x_maxpool, conv_out_shape2, \
           pos_maxpool_pos, x_conv2, conv2_input, x_conv, cost


# ################################################################################
# BACKWARD PASS
# ################################################################################
def backward(scores, one_hot_encoding_labels, fc2_input, fc1, fc1_input, x_maxpool, conv_out_shape2,
             pos_maxpool_pos, x_conv2, conv2_input, x_conv, input_data,
             kernel, kernel2, fc1_w, fc1_b, fc2_w, fc2_b):
    delta_2 = (scores - one_hot_encoding_labels)  # / BATCH_SIZE  # TODO: check this
    d_fc2_w = delta_2 @ fc2_input.T
    d_fc2_b = np.sum(delta_2, axis=1, keepdims=True)

    delta_1 = np.multiply(fc2_w.T @ delta_2, dReLU(fc1))
    d_fc1_w = delta_1 @ fc1_input.T
    d_fc1_b = np.sum(delta_1, axis=1, keepdims=True)

    # gradient WRT x0
    # delta_0 = np.multiply(fc1_w.T @ delta_1, 1.0)
    delta_0 = fc1_w.T @ delta_1

    # unflatten operation
    delta_maxpool = delta_0.reshape(x_maxpool.shape)

    # gradients through the maxpool operation

    delta_conv2 = fast_maxpool_backprop(
        delta_maxpool,
        conv_out_shape2,
        padding=0,  # not working with max pool padding, seems to be related to the pooling padding
        stride=2,
        max_pool_size=2,
        pos_result=pos_maxpool_pos)

    dX1 = np.multiply(delta_conv2, dReLU(x_conv2))

    conv2_delta, dX2 = fast_convolution_backprop(conv2_input, kernel2, dX1, padding=0)

    dX2 = np.multiply(dX2, dReLU(x_conv))

    conv1_delta, _ = fast_convolution_backprop(input_data, kernel, dX2, padding=0)

    return delta_2


def softmax_cost(y, y_hat):
    return -np.sum(y * np.log(y_hat), axis=0)


def gradientCheck(input_data, input_labels, epsilon=1e-7):
    # One hot encoding
    one_hot_encoding_labels = np.zeros((input_labels.size, 10))
    for i in range(input_labels.shape[0]):
        position = input_labels[i]
        one_hot_encoding_labels[i, position] = 1
    one_hot_encoding_labels = one_hot_encoding_labels.T

    kernel = generate_kernel(input_channels=3, output_channels=8, kernel_h=3, kernel_w=3)
    kernel2 = generate_kernel(input_channels=8, output_channels=16, kernel_h=3, kernel_w=3)

    fc1_w = np.random.randn(64, 3136) / np.sqrt(3136 / 2)
    fc1_b = np.zeros((64, 1))

    fc2_w = np.random.randn(10, 64) / np.sqrt(64 / 2)
    fc2_b = np.zeros((10, 1))

    scores, fc2_input, fc1, fc1_input, x_maxpool, conv_out_shape2, \
        pos_maxpool_pos, x_conv2, conv2_input, x_conv, cost \
        = forward(input_data, input_labels, kernel, kernel2,
                  fc1_w, fc1_b, fc2_w, fc2_b, one_hot_encoding_labels)

    delta_2 = backward(scores, one_hot_encoding_labels, fc2_input, fc1, fc1_input, x_maxpool, conv_out_shape2, \
                       pos_maxpool_pos, x_conv2, conv2_input, x_conv, input_data, \
                       kernel, kernel2, fc1_w, fc1_b, fc2_w, fc2_b)

    Ascores, Afc2_input, Afc1, Afc1_input, Ax_maxpool, Aconv_out_shape2, \
    Apos_maxpool_pos, Ax_conv2, Aconv2_input, Ax_conv, Acost \
        = forward(input_data, input_labels,
                  kernel + epsilon, kernel2 + epsilon,
                  fc1_w + epsilon, fc1_b + epsilon,
                  fc2_w + epsilon, fc2_b + epsilon, one_hot_encoding_labels)

    Bscores, Bfc2_input, Bfc1, Bfc1_input, Bx_maxpool, Bconv_out_shape2, \
    Bpos_maxpool_pos, Bx_conv2, Bconv2_input, Bx_conv, Bcost \
        = forward(input_data, input_labels,
                  kernel - epsilon, kernel2 - epsilon,
                  fc1_w - epsilon, fc1_b - epsilon,
                  fc2_w - epsilon, fc2_b - epsilon, one_hot_encoding_labels)

    approx = (Acost - Bcost) / (2. * epsilon)

    # scores = softmax result
    # to compare with delta_2
    delta2_flat0 = (delta_2 / 128).flat[0]
    delta2_flat1 = (delta_2 / 128).flat[1]
    print("Check Passed: " + str(np.isclose(approx, delta_2)))

    print()
