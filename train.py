import random
import numpy as np
from caffe2.python.helpers.pooling import max_pool

from cifar10 import Cifar10
from convolution import *
from convolution_to_delete import *
from flatten import flatten
from max_pooling import *
from relu import ReLU
from softmax import *
from utils import *
from cross_entropy import *


def show_test_image(images, labels, classes):
    # test_max_pool()
    # test_convolution()
    test_softmax()

    # Kernel for the convolution layer
    kernel = init_random_kernel()

    input_data = np.asarray([images[0], images[1]])
    input_labels = np.asarray([labels[0], labels[1]])

    # One hot encoding
    one_hot_encoding_labels = np.zeros((input_labels.size, 10))
    for i in range(input_labels.shape[0]):
        position = int(input_labels[i])
        one_hot_encoding_labels[i, position] = 1
    one_hot_encoding_labels = one_hot_encoding_labels.T

    # ####################
    # Forward Pass
    # ####################
    x = convolve_2d(input_data, kernel)
    a = x.shape
    x = ReLU(x)
    x, pos_maxpool_result = maxPool(x)
    x = flatten(x)

    # First fc layer
    fc1_w = np.random.rand(60, 450) / np.sqrt(450)
    fc1_b = np.zeros((60, 1)) / np.sqrt(60)
    fc1_output = np.matmul(fc1_w, x) + fc1_b

    x = ReLU(fc1_output)
    a = x.shape

    # Second fc layer
    fc2_w = np.random.rand(10, 60) / np.sqrt(60)
    fc2_b = np.zeros((10, 1)) / np.sqrt(60)
    fc2_output = np.matmul(fc2_w, x) + fc2_b

    # Finally apply the softmax
    scores = softmax(fc2_output)

    # Compute the cross entropy loss
    ce = cross_entropy(scores, one_hot_encoding_labels)

    predicted_class = np.argmax(scores, axis=0)
    print(scores.shape)
    print('predicted class: {}'.format(predicted_class))
    print()

    # ####################
    # Backward Pass
    # ####################

    # Start computing the derivatives required from the backpropagation algorithm
    # The 1st derivative is the one related to the softmax.
    # The softmax is a vector, therefore we have to compute the Jacobian.
    # In each cell of the Jacobian we have the partial derivative of the i-th output WRT the j-th input.

    # The input of the softmax is the fc2_output
    # The output of the softmax is a vector, whose element sum up to one.

    delta_2 = (scores - one_hot_encoding_labels)
    a = delta_2.shape
    db2 = np.sum(delta_2, axis=1, keepdims=True)
    a = db2.shape
    dW2 = delta_2 @ x.T
    a = dW2.shape
    print()


def main():
    dataset = Cifar10()

    # show_test_image(dataset.train_images, dataset.train_labels, dataset.classes)

    # show_test_image()
    # max_pool_backprop_test()
    # backprop_test()

    convolution_comparisons()


if __name__ == '__main__':
    main()
