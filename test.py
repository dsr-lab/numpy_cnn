import numpy as np

from convolution import fast_convolve_2d, convolve_2d
from cross_entropy import cross_entropy
from flatten import flatten
from max_pooling import fast_max_pool, max_pool
from relu import ReLU
from softmax import softmax
from utils import *
from timeit import default_timer as timer


def load_weights():
    print()
    return None, None, None, None, None, None


def test(test_images, test_labels, use_fast_conv, batch_size, padding):

    kernel, kernel2, fc2_w, fc2_b, fc1_w, fc1_b = load_weights()

    # Divide the datasets into batches
    test_images_batches = np.split(test_images, np.arange(batch_size, len(test_images), batch_size))
    test_images_labels = np.split(test_labels, np.arange(batch_size, len(test_labels), batch_size))

    start = timer()

    test_batch_loss = 0
    test_batch_acc = 0
    test_samples = 0

    for idx, input_data in enumerate(test_images_batches):
        test_samples += input_data.shape[0]

        input_labels = test_images_labels[idx]

        # One hot encoding
        one_hot_encoding_labels = np.zeros((input_labels.size, 10))
        for i in range(input_labels.shape[0]):
            position = input_labels[i]
            one_hot_encoding_labels[i, position] = 1

        # ################################################################################
        # FORWARD PASS
        # ################################################################################

        # ********************
        # CONV 1 + RELU
        # ********************
        if use_fast_conv:
            x_conv = fast_convolve_2d(input_data, kernel, padding=padding)
        else:
            x_conv = convolve_2d(input_data, kernel, padding=padding)

        conv2_input = ReLU(x_conv)

        # ********************
        # CONV 2 + RELU
        # ********************
        if use_fast_conv:
            x_conv2 = fast_convolve_2d(conv2_input, kernel2, padding=padding)
        else:
            x_conv2 = convolve_2d(conv2_input, kernel2, padding=padding)

        maxpool_input = ReLU(x_conv2)

        # ********************
        # MAXPOOL
        # ********************
        if use_fast_conv:
            x_maxpool, pos_maxpool_pos = fast_max_pool(maxpool_input)
        else:
            x_maxpool, pos_maxpool_pos = max_pool(maxpool_input)

        # ********************
        # FLATTEN + FCs
        # ********************
        fc1_input = flatten(x_maxpool)

        # First fc layer
        fc1 = np.matmul(fc1_input, fc1_w) + fc1_b
        fc2_input = ReLU(fc1)

        # Second fc layer
        fc2 = np.matmul(fc2_input, fc2_w) + fc2_b

        # Apply the softmax for computing the scores
        scores = softmax(fc2)

        # Compute the cross entropy loss
        ce = cross_entropy(scores, one_hot_encoding_labels) * input_data.shape[0]

        # Compute prediction and accuracy
        acc = accuracy(scores, input_labels) * input_data.shape[0]

        # compute for the entire epoch!
        test_batch_acc += acc
        test_batch_loss += ce

    end = timer()

    print('TRAIN Accuracy: {:.3f}\tTRAIN Loss: {:.3f}'.
          format(test_batch_acc / test_samples, test_batch_loss / test_samples))
    print("Elapsed time (s): {}".format(end - start))
    print()
