from plots.plot_utils import *


def show_naive_fast_comparison_plots(naive_file_path, fast_file_path):

    fig, (ax1) = plt.subplots(1, 1)
    #fig.set_figwidth(20)
    #fig.set_figheight(20)

    plt.text(x=0.5, y=0.94, s=f"Naive VS Fast implementation", fontsize=18,
             ha="center", transform=fig.transFigure)
    plt.text(x=0.5, y=0.88, s=f"Comparison on CIFAR 10 (small)", fontsize=14,
             weight="bold", ha="center", transform=fig.transFigure)

    _, _, _, _, _, naive_time = get_train_val_values(naive_file_path)
    _, _, _, _, _, fast_time = get_train_val_values(fast_file_path)

    a = naive_time[0]
    b = fast_time[0]

    x = np.arange(2)
    barlist = ax1.bar(x, height=[naive_time[0], fast_time[0]], width=0.5)
    barlist[1].set_color('r')
    plt.subplots_adjust(top=0.8, wspace=0.3)
    plt.xticks(x, ['Naive', 'Fast'])

    ax1.set_xlabel('Type', weight="bold")
    ax1.set_ylabel('Seconds per EPOCH', weight="bold")
    plt.show(block=True)
