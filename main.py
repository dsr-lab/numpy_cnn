from dataset_cifar10 import Cifar10
from dataset_mnist import Mnist
from model import *
from plots.plot import show_result_plots


def main():
    if USE_CIFAR_10:
        dataset = Cifar10()
    else:
        dataset = Mnist()

    train_images, train_labels, \
    validation_images, validation_labels, \
    test_images, test_labels = \
        dataset.get_small_datasets() if TRAIN_SMALL_DATASET else dataset.get_datasets()

    show_result_plots()

    # convolution_method_comparisons(train_images[:10], generate_kernel(), np.random.rand(10, 16, 30, 30))

    # ######################################################################
    # TRAIN + VALIDATION
    # ######################################################################
    # train_model(train_images, train_labels,
    #             validation_images, validation_labels, EPOCHS)

    # ######################################################################
    # TRAIN (on full train set)
    # ######################################################################
    # Load the number of epochs obtained after running the model on train
    # and validation set
    # _, epochs = load_weights(VALIDATION_WEIGHTS_PATH)
    #
    # print(f'Best epochs loaded from file system: {epochs[0] - 1}')
    # print()
    #
    # train_model(np.concatenate((train_images, validation_images)),
    #             np.concatenate((train_labels, validation_labels)),
    #             None, None, epochs[0])

    # ######################################################################
    # TEST
    # ######################################################################
    test_model(TRAIN_WEIGHTS_PATH, test_images, test_labels, dataset.classes)

    print()


if __name__ == '__main__':
    main()
