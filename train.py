import random
import numpy as np
from caffe2.python.helpers.pooling import max_pool

from cifar10 import Cifar10
from convolution import *
from flatten import flatten
from max_pooling import *
from relu import ReLU
from softmax import *
from utils import *


def show_test_image(images, labels, classes):
    idx = random.randint(0, 10000)
    selected_class_idx = int(labels[idx])

    #test_max_pool()
    #test_convolution()
    test_softmax()

    # Kernel for the convolution layer
    kernel = init_random_kernel()

    input_data = np.asarray([images[0], images[2]])

    x = convolve_2d(input_data, kernel)
    x = ReLU(x)
    x = maxPool(x)
    x = flatten(x)

    # First fc layer
    fc1_w = np.random.rand(60, 450) / np.sqrt(450)
    fc1_b = np.zeros((60, 1)) / np.sqrt(60)
    fc1_output = np.matmul(fc1_w, x) + fc1_b

    x = ReLU(fc1_output)

    # Second fc layer
    fc2_w = np.random.rand(10, 60) / np.sqrt(60)
    fc2_b = np.zeros((10, 1)) / np.sqrt(60)
    fc2_output = np.matmul(fc2_w, x) + fc2_b

    # Finally apply the softmax
    scores = softmax(fc2_output)
    predicted_class = np.argmax(scores, axis=0)
    print(scores.shape)
    print('predicted class: {}'.format(predicted_class))
    print()


def main():
    dataset = Cifar10()
    print(dataset.train_images[0].shape)
    show_test_image(dataset.train_images, dataset.train_labels, dataset.classes)



if __name__ == '__main__':
    main()
