import random
import numpy as np

from cifar10 import Cifar10
from convolution import test_convolution
from utils import show_image



def show_test_image(images, labels, classes):
    idx = random.randint(0, 10000)
    selected_class_idx = int(labels[idx])

    test_convolution()


    print('Selected index: {} - Class: {}'
          .format(idx, classes[selected_class_idx]))

    img = images[idx]
    show_image(img)


def main():
    dataset = Cifar10()
    print(dataset.train_images[0].shape)
    show_test_image(dataset.train_images, dataset.train_labels, dataset.classes)


if __name__ == '__main__':
    main()
