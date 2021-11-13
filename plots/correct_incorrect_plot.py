from plots.plot_utils import *


def correct_incorrect_plot(correct, incorrect, dataset_classes_desc):
    correct_images, correct_true_labels, _ = correct
    incorrect_images, incorrect_true_labels, incorrect_predicted_labels = incorrect

    fig, (ax1, ax2) = plt.subplots(2, 5)  # , constrained_layout=True)
    fig.set_figwidth(20)
    fig.set_figheight(20)

    plt.text(x=0.5, y=0.95,
             s=f"Predictions comparison",
             fontsize=20, ha="center", transform=fig.transFigure
             )
    plt.text(x=0.5, y=0.89, s=f"Correct Predictions", fontsize=15, weight="bold", ha="center",
             transform=fig.transFigure)
    plt.text(x=0.5, y=0.47, s=f"Incorrect Predictions", fontsize=15, weight="bold", ha="center",
             transform=fig.transFigure)

    # Correctly classified samples
    for idx, ax in enumerate(ax1):
        image = correct_images[idx]
        __ax_plot_image(ax, image)

        ax.set_title(dataset_classes_desc[correct_true_labels[idx]])
        ax.axis('off')

    # Incorrectly classified samples
    for idx, ax in enumerate(ax2):
        ax.axis('off')
        if idx >= len(incorrect_images):
            continue
        image = incorrect_images[idx]
        __ax_plot_image(ax, image)
        ax.set_title(
            f'Predicted: {dataset_classes_desc[incorrect_predicted_labels[idx]]}'
            f'\nTrue: {dataset_classes_desc[incorrect_true_labels[idx]]}'
        )

    plt.subplots_adjust(top=0.9, wspace=0.3)
    plt.show(block=True)

    print()


def __ax_plot_image(ax, image):
    image = normalize_image_0_to_1(image)

    if image.shape[0] == 1:
        image = image.squeeze(axis=0)
        ax.imshow(image, cmap='gray')
    else:
        ax.imshow(np.transpose(image, (1, 2, 0)))
