import numpy as np


def maxpool_multiple(input_image, stride=2):
    input_width = input_image.shape[3]
    input_height = input_image.shape[2]
    filter_width = 2
    filter_height = 2

    output_width = int((input_width - filter_width) / stride) + 1
    output_height = int((input_height - filter_height) / stride) + 1

    output_image = np.zeros((input_image.shape[0], input_image.shape[1], output_width, output_height))
    for i in range(output_image.shape[0]):
        output_image[i:i + 1, :, :, :] = maxpool(input_image[i:i + 1, :, :, :], stride=2)
    return output_image


def maxpool(input_image, stride=2):
    input_width = input_image.shape[3]
    input_height = input_image.shape[2]
    filter_width = 2
    filter_height = 2
    n_channels = input_image.shape[1]
    num_images = input_image.shape[0]

    output_width = int((input_width - filter_width) / stride) + 1
    output_height = int((input_height - filter_height) / stride) + 1
    output = np.zeros((n_channels, output_width * output_height))
    c = 0
    for height in range(0, input_height, stride):
        if height + filter_height <= input_height:
            image_rectangle = input_image[0, :, height:height + filter_height, :]
            for width in range(0, input_width, stride):
                if width + filter_width <= input_width:
                    image_square = image_rectangle[:, :, width:width + filter_width]
                    image_flatten = image_square.reshape(-1, 1)
                    #                     print(image_flatten)
                    #                     print('----')
                    output[:, c:c + 1] = np.array([float(max(i)) for i in np.split(image_flatten, n_channels)]).reshape(
                        -1, 1)
                    c += 1

    final_output = np.array(np.hsplit(output, 1)).reshape((1, n_channels, output_height, output_width))

    return final_output


def maxpool_indices(input_image, stride=2, filter_height=2, filter_width=2):
    positional_vector = []

    for channel in range(input_image.shape[1]):
        x = -1

        chosen_image_channel = input_image[:, channel, :, :]
        for height in range(0, chosen_image_channel.shape[1], stride):
            if height + stride <= chosen_image_channel.shape[1]:
                image_rectangle = chosen_image_channel[:, height:height + filter_height, :]
                x = x + 1
                y = -1
                # print('Value of x:',x)
                for width in range(0, image_rectangle.shape[2], stride):
                    if width + stride <= image_rectangle.shape[2]:
                        y = y + 1
                        # print('Value of y:',y)
                        image_square = image_rectangle[:, :, width:width + filter_width]

                        a, b, c = np.unravel_index(image_square.argmax(), image_square.shape)

                        positional_vector.append([0, channel, int(b) + height, int(c) + width, 0, channel, x, y])
    return positional_vector


def maxpool_indices_multiple(input_image, stride=2, filter_height=2, filter_width=2):
    positional_vector = []
    for i in range(input_image.shape[0]):
        positional_vector.append(
            maxpool_indices(input_image[i:i + 1, :, :, :], stride=2, filter_height=2, filter_width=2))
    return positional_vector
