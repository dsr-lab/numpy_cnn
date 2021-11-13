from plots.plot_utils import *


def __create_subplot(ax1, x, show_title=True):
    train_accuracy, train_loss, validation_accuracy, validation_loss, epochs, _ = x

    if show_title:
        ax1[0].set_title('ACCURACY')
    ax1[0].plot(epochs, train_accuracy, label='TRAIN')
    ax1[0].plot(epochs, validation_accuracy, label='VALID')
    ax1[0].set_xlabel('Epochs')
    ax1[0].set_ylabel('Value')

    offset = 10
    min_value = np.min((np.min(train_accuracy), np.min(validation_accuracy)))
    max_value = np.max((np.max(train_accuracy), np.max(validation_accuracy)))
    delta = offset * (max_value - min_value) / 100

    ax1[0].set_ylim([min_value-delta, max_value+delta])
    ax1[0].legend()

    if show_title:
        ax1[1].set_title('LOSS')
    ax1[1].plot(epochs, train_loss, label='TRAIN')
    ax1[1].plot(epochs, validation_loss, label='VALID')
    ax1[1].set_xlabel('Epochs')
    ax1[1].set_ylabel('Value')

    min_value = np.min((np.min(train_loss), np.min(validation_loss)))
    max_value = np.max((np.max(train_loss), np.max(validation_loss)))
    delta = offset * (max_value - min_value) / 100
    ax1[1].set_ylim([min_value-delta, max_value+delta])
    ax1[1].legend()


def show_optimizer_comparison_plots(file_path_1, file_path_2, file_path_3):

    title_array1 = get_file_name_from_path(file_path_1)
    title_array2 = get_file_name_from_path(file_path_2)
    title_array3 = get_file_name_from_path(file_path_3)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 2)  # , constrained_layout=True)

    plt.text(x=0.5, y=0.95, s=f"Optimizer comparison on {title_array1[0]}", fontsize=20, ha="center", transform=fig.transFigure)
    plt.text(x=0.5, y=0.91, s=f"{title_array1[1]}", fontsize=14, weight="bold", ha="center", transform=fig.transFigure)
    plt.text(x=0.5, y=0.60, s=f"{title_array2[1]}", fontsize=14, weight="bold", ha="center", transform=fig.transFigure)
    plt.text(x=0.5, y=0.29, s=f"{title_array3[1]}", fontsize=14, weight="bold", ha="center", transform=fig.transFigure)

    fig.set_figwidth(20)
    fig.set_figheight(20)

    __create_subplot(ax1, get_train_val_values(file_path_1), show_title=True)
    __create_subplot(ax2, get_train_val_values(file_path_2))
    __create_subplot(ax3, get_train_val_values(file_path_3))

    plt.subplots_adjust(top=0.88, wspace=0.3, hspace=0.5, bottom=0.05)
    plt.show(block=True)

