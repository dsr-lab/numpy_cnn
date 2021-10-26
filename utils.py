import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("TkAgg")


def show_gray_scale_image(image, title=None):
    a = image.shape
    # initially (3, 32, 32), pyplot expects (32, 32, 3)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show(block=True)

    if title is not None:
        plt.title(title)


def show_image(image, title=None):
    # npimg = image.numpy()

    # initially (3, 32, 32), pyplot expects (32, 32, 3)
    plt.imshow(np.transpose(image, (1, 2, 0)))
    plt.axis('off')
    plt.show(block=True)

    if title is not None:
        plt.title(title)


def accuracy(scores, labels):
    n_samples = scores.shape[1]

    predictions = np.argmax(scores, axis=0)

    n_correct = (labels == predictions).sum()

    acc = n_correct / n_samples

    return acc

