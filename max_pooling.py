import numpy as np
from PIL import Image

from utils import *


def maxPool(input_images, stride=2, filter_h=2, filter_w=2, padding=0):
    # Retrieve the input size
    input_h = input_images.shape[2]
    input_w = input_images.shape[3]

    # Compute the expected output size (h,w)
    output_h = int(((input_h + 2 * padding - filter_h) / stride) + 1)
    output_w = int(((input_w + 2 * padding - filter_w) / stride) + 1)

    # Init the maxpool matrix result with zero values
    maxpool_result = np.zeros((
        input_images.shape[0],
        input_images.shape[1],
        output_h,
        output_w
    ))

    pos_result = []

    # Cycle all the images in the batch
    for i in range(maxpool_result.shape[0]):
        current_image = input_images[i, :, :, :]
        current_image = np.pad(current_image, ((0, 0), (padding, padding), (padding, padding)), mode='constant')
        single_maxpool_result, single_pos_vector = \
            __process_single_image(current_image, stride, output_h, output_w, filter_h, filter_w)
        maxpool_result[i, :, :, :] = single_maxpool_result
        pos_result.append(single_pos_vector)
    return maxpool_result, pos_result


def maxpool_backprop(gradient_values, pos_result, conv_shape):
    delta_conv = np.zeros(conv_shape)
    delta_conv_shape = delta_conv.shape
    for image in range(len(pos_result)):
        indices = pos_result[image]
        for p in indices:
            '''
                0) original image channel
                1) original image row
                2) original image column
                3) maxpooled row
                4) maxpooled column
            '''
            a = p[0]
            b = p[1]
            c = gradient_values[image, p[0], p[3], p[4]]

            delta_conv[image, p[0], p[1], p[2]] = gradient_values[image, p[0], p[3], p[4]]
    return delta_conv


def __process_single_image(image, stride, output_h, output_w, filter_h, filter_w):
    # Init the maxpool matrix result with zero values
    maxpool_result = np.zeros((
        image.shape[0],
        output_h,
        output_w
    ))

    pos_vector = []
    '''
        1) original image channel
        2) original image row
        3) original image column
        5) maxpooled row
        6) maxpooled column
    '''

    # Cycle all the channels
    for channel in range(0, image.shape[0]):
        # height index for the output activation
        output_h_idx = 0

        for height in range(0, image.shape[1], stride):
            # width index for the output activation
            output_w_idx = 0

            # get a portion of the image
            image_rectangle = image[channel, height:height + filter_h, :]
            image_rectangle_shape = image_rectangle.shape
            if image_rectangle.shape[0] < filter_h:
                continue
            else:
                for width in range(0, image_rectangle.shape[1], stride):
                    image_portion = image_rectangle[:, width:width + filter_w]
                    if image_portion.shape[1] < filter_w:
                        continue
                    else:
                        # get the indexes where the maximum value has been found
                        row, column = np.unravel_index(image_portion.argmax(), image_portion.shape)

                        pos_vector.append([channel, row + height, column + width, output_h_idx, output_w_idx])

                        # Perform the max pooling
                        maxpool_result[channel, output_h_idx, output_w_idx] = \
                            np.max(image_portion)
                        output_w_idx += 1
                output_h_idx += 1

    return maxpool_result, pos_vector
