import os

from plots.initializer_comparison_plot import show_initializer_comparison_plots
from plots.optimizer_comparison_plot import show_optimizer_comparison_plots


def __optimizer_comparison():
    base_path = os.path.join('plots', 'results', 'optimizer_comparison')

    file_name1 = os.path.join(base_path, 'MNIST_ADAM')
    file_name2 = os.path.join(base_path, 'MNIST_MOMENTUM')
    file_name3 = os.path.join(base_path, 'MNIST_SGD')
    show_optimizer_comparison_plots(file_name1, file_name2, file_name3)

    file_name1 = os.path.join(base_path, 'CIFAR10_ADAM')
    file_name2 = os.path.join(base_path, 'CIFAR10_MOMENTUM')
    file_name3 = os.path.join(base_path, 'CIFAR10_SGD')
    show_optimizer_comparison_plots(file_name1, file_name2, file_name3)


def __initializer_comparison():
    base_path = os.path.join('plots', 'results', 'initializer_comparison')

    cifar10_adam_he = os.path.join(base_path, 'CIFAR10_ADAM_HE')
    cifar10_adam_xavier = os.path.join(base_path, 'CIFAR10_ADAM_XAVIER')
    cifar10_momentum_he = os.path.join(base_path, 'CIFAR10_MOMENTUM_HE')
    cifar10_momentum_xavier = os.path.join(base_path, 'CIFAR10_MOMENTUM_XAVIER')
    cifar10_sgd_he = os.path.join(base_path, 'CIFAR10_SGD_HE')
    cifar10_sgd_xavier = os.path.join(base_path, 'CIFAR10_SGD_XAVIER')

    show_initializer_comparison_plots(
        cifar10_adam_he, cifar10_adam_xavier,
        cifar10_momentum_he, cifar10_momentum_xavier,
        cifar10_sgd_he, cifar10_sgd_xavier
    )


def show_result_plots():
    __optimizer_comparison()
    __initializer_comparison()

    print()

