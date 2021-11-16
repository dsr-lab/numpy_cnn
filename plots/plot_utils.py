import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import re


def get_train_val_values(file_name, file_ext='.txt'):
    epochs = []
    epochs_time = []
    train_accuracy = []
    validation_accuracy = []

    train_loss = []
    validation_loss = []

    idx = 0

    with open(file_name + file_ext) as file:
        for line in file:
            if 'TRAIN Accuracy:' in line:
                epochs.append(idx)
                idx += 1
                accuracy = re.search('TRAIN Accuracy:(.*)TRAIN Loss:', line).group(1).strip()
                loss = re.search('TRAIN Loss:(.*)', line).group(1).strip()

                train_accuracy.append(float(accuracy))
                train_loss.append(float(loss))

            elif 'VALID Accuracy:' in line:
                accuracy = re.search('VALID Accuracy:(.*)VALID Loss:', line).group(1).strip()
                loss = re.search('VALID Loss:(.*)', line).group(1).strip()

                validation_accuracy.append(float(accuracy))
                validation_loss.append(float(loss))
            elif 'completed in (s):' in line:
                time = re.search('completed in \(s\):(.*)', line).group(1).strip()

                epochs_time.append(float(time))

    return train_accuracy, train_loss, validation_accuracy, validation_loss, epochs, epochs_time


def get_file_name_from_path(path):
    return os.path.basename(path).split('_')


def show_gray_scale_image(image, title=None):
    image = image.squeeze(axis=0)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show(block=True)

    if title is not None:
        plt.title(title)


def show_first_layer(image, title):
    fig = plt.figure(figsize=(10, 10))
    columns = image.shape[0]
    rows = 1
    for i in range(0, columns * rows):
        fig.add_subplot(rows, columns, i + 1)
        img = normalize_image_0_to_1(image[i])
        img = np.transpose(img, (1, 2, 0))
        plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.savefig(f'test/{title}.png')
    plt.close(fig)
    # plt.show(block=True)


def normalize_image_0_to_1(image):
    if image.min() < 0:
        return np.interp(image, (image.min(), image.max()), (0, 1))
    return image


def show_image_from_file(path, title, block=True):
    img = mpimg.imread(path)

    plt.figure(figsize=(20, 20))
    plt.title(title, fontdict={'fontsize': 20, 'fontweight': 'bold'})
    plt.imshow(img)
    plt.axis('off')
    plt.show(block=block)


def show_image(image, title=None):
    image = normalize_image_0_to_1(image)
    # npimg = image.numpy()

    # initially (3, 32, 32), pyplot expects (32, 32, 3)
    plt.imshow(np.transpose(image, (1, 2, 0)))
    plt.axis('off')
    plt.show(block=True)

    if title is not None:
        plt.title(title)
