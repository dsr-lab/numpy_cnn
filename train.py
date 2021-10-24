import random
import numpy as np
from caffe2.python.helpers.pooling import max_pool

from cifar10 import Cifar10
from convolution import *
from convolution_to_delete import *
from flatten import flatten
from max_pooling import *
from relu import ReLU, dReLU
from softmax import *
from utils import *
from cross_entropy import *

BATCH_SIZE = 128
EPOCHS = 20


def train_network(train_images, train_labels,
                  test_images, test_labels,
                  valid_images, valid_labels):
    # test_max_pool()
    # test_convolution()
    # test_softmax()

    # TODO: optimize this without creating a copy of the dataset
    train_images_batches = np.split(train_images, np.arange(BATCH_SIZE, len(train_images), BATCH_SIZE))
    train_images_labels = np.split(train_labels, np.arange(BATCH_SIZE, len(train_labels), BATCH_SIZE))

    # Kernel for the convolution layer
    kernel = init_random_kernel()

    # fc1_w = np.random.rand(60, 450) / np.sqrt(450)
    # fc1_b = np.zeros((60, 1)) / np.sqrt(450)
    fc1_stdv = 1. / np.sqrt(450)
    fc1_w = np.random.uniform(low=-fc1_stdv, high=fc1_stdv, size=(60, 450))
    fc1_b = np.random.uniform(low=-fc1_stdv, high=fc1_stdv, size=(60, 1))

    # fc2_w = np.random.rand(10, 60) / np.sqrt(60)
    # fc2_b = np.zeros((10, 1)) / np.sqrt(60)
    fc2_stdv = 1. / np.sqrt(60)
    fc2_w = np.random.uniform(low=-fc2_stdv, high=fc2_stdv, size=(10, 60))
    fc2_b = np.random.uniform(low=-fc2_stdv, high=fc2_stdv, size=(10, 1))

    learning_rate = 0.001

    for e in range(EPOCHS):

        batch_loss = 0
        batch_acc = 0

        for idx, input_data in enumerate(train_images_batches):

            input_labels = train_images_labels[idx]

            # input_data = np.asarray([train_images[0], images[1]])
            # input_labels = np.asarray([labels[0], labels[1]])

            # One hot encoding
            one_hot_encoding_labels = np.zeros((input_labels.size, 10))
            for i in range(input_labels.shape[0]):
                position = int(input_labels[i])
                one_hot_encoding_labels[i, position] = 1
            one_hot_encoding_labels = one_hot_encoding_labels.T

            # ####################
            # Forward Pass
            # ####################
            x_conv = convolve_2d(input_data, kernel)
            conv_out_shape = x_conv.shape
            x = ReLU(x_conv)
            x_maxpool, pos_maxpool_pos = maxPool(x)
            x_flatten = flatten(x_maxpool)

            # First fc layer
            fc1 = np.matmul(fc1_w, x_flatten) + fc1_b
            fc1 = ReLU(fc1)

            # Second fc layer
            fc2 = np.matmul(fc2_w, fc1) + fc2_b

            # Finally apply the softmax
            scores = softmax(fc2)

            # Compute the cross entropy loss
            ce = cross_entropy(scores, one_hot_encoding_labels)

            # Compute prediction and accuracy
            predictions = np.argmax(scores, axis=0)
            acc = accuracy(predictions, input_labels)

            # compute for the entire epoch!
            batch_acc += acc
            batch_loss += ce

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
            d_fc2_w = delta_2 @ fc1.T
            d_fc2_b = np.sum(delta_2, axis=1, keepdims=True)

            delta_1 = np.multiply(fc2_w.T @ delta_2, dReLU(fc1_w @ x_flatten + fc1_b))
            d_fc1_w = delta_1 @ x_flatten.T
            d_fc1_b = np.sum(delta_1, axis=1, keepdims=True)

            delta_0 = np.multiply(fc1_w.T @ delta_1, 1.0)

            delta_maxpool = delta_0.reshape(x_maxpool.shape)
            delta_conv = maxpool_backprop(delta_maxpool, pos_maxpool_pos, conv_out_shape)

            delta_conv = np.multiply(delta_conv, dReLU(x_conv))

            conv1_delta = convolution_backprop(input_data, kernel, delta_conv)
            a = conv1_delta.shape

            fc2_w = fc2_w - learning_rate * d_fc2_w
            fc2_b = fc2_b - learning_rate * d_fc2_b

            fc1_w = fc1_w - learning_rate * d_fc1_w
            fc1_b = fc1_b - learning_rate * d_fc1_b

            kernel = kernel - learning_rate * conv1_delta

        print('Epoch: {} - Accuracy: {} - Loss: {}'.format(e, batch_acc, batch_loss))



def main():
    dataset = Cifar10()

    train_images, train_labels, \
    validation_images, validation_labels, \
    test_images, test_labels = dataset.get_small_datasets()

    a = train_labels
    print()

    train_network(train_images, train_labels, validation_images, validation_labels, test_images, test_labels)
    #max_pool_backprop_test()

    # test_max_pool()
    # show_test_image()
    # max_pool_backprop_test()
    # backprop_test()

    # convolution_method_comparisons()


if __name__ == '__main__':
    main()
