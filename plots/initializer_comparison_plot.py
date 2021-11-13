from plots.plot_utils import *


def __create_subplot(ax1, he, xavier, show_title=True):

    train_accuracy_he, train_loss_he, _, _, epochs, _ = he
    train_accuracy_xavier, train_loss_xavier, _, _, epochs, _ = xavier

    if show_title:
        ax1[0].set_title('ACCURACY')

    ax1[0].plot(epochs, train_accuracy_he, label='HE')
    ax1[0].plot(epochs, train_accuracy_xavier, label='XAVIER')
    ax1[0].set_xlabel('Epochs')
    ax1[0].set_ylabel('Value')

    offset = 10
    min_value = np.min((np.min(train_accuracy_he), np.min(train_accuracy_xavier)))
    max_value = np.max((np.max(train_accuracy_he), np.max(train_accuracy_xavier)))
    delta = offset * (max_value-min_value) / 100

    ax1[0].set_ylim([min_value-delta, max_value+delta])
    ax1[0].legend()

    if show_title:
        ax1[1].set_title('LOSS')

    ax1[1].plot(epochs, train_loss_he, label='HE')
    ax1[1].plot(epochs, train_loss_xavier, label='XAVIER')
    ax1[1].set_xlabel('Epochs')
    ax1[1].set_ylabel('Value')

    min_value = np.min((np.min(train_loss_he), np.min(train_loss_xavier)))
    max_value = np.max((np.max(train_loss_he), np.max(train_loss_xavier)))
    delta = offset * (max_value - min_value) / 100

    ax1[1].set_ylim([min_value-delta, max_value+delta])
    ax1[1].legend()


def show_initializer_comparison_plots(
        cifar10_adam_he, cifar10_adam_xavier,
        cifar10_momentum_he, cifar10_momentum_xavier,
        cifar10_sgd_he, cifar10_sgd_xavier):

    title_array1 = get_file_name_from_path(cifar10_adam_he)
    title_array2 = get_file_name_from_path(cifar10_momentum_he)
    title_array3 = get_file_name_from_path(cifar10_sgd_he)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 2)  # , constrained_layout=True)

    plt.text(x=0.5, y=0.95,
             s=f"Weight initialization comparison on {title_array1[0]} (small)",
             fontsize=20, ha="center", transform=fig.transFigure
             )
    plt.text(x=0.5, y=0.91, s=f"{title_array1[1]}", fontsize=14, weight="bold", ha="center", transform=fig.transFigure)
    plt.text(x=0.5, y=0.60, s=f"{title_array2[1]}", fontsize=14, weight="bold", ha="center", transform=fig.transFigure)
    plt.text(x=0.5, y=0.29, s=f"{title_array3[1]}", fontsize=14, weight="bold", ha="center", transform=fig.transFigure)

    fig.set_figwidth(20)
    fig.set_figheight(20)

    __create_subplot(ax1, get_train_val_values(cifar10_adam_he), get_train_val_values(cifar10_adam_xavier))
    __create_subplot(ax2, get_train_val_values(cifar10_momentum_he), get_train_val_values(cifar10_momentum_xavier))
    __create_subplot(ax3, get_train_val_values(cifar10_sgd_he), get_train_val_values(cifar10_sgd_xavier))

    plt.subplots_adjust(top=0.88, wspace=0.3, hspace=0.5, bottom=0.05)
    plt.show(block=True)
