import numpy as np
from PIL import Image

from utils import *


def test_max_pool():
    '''
    data     = [
        [[3, 1, 7, 2, 5], [5, 1, 0, 9, 2], [8, 2, 4, 9, 3], [4, 3, 1, 1, 4]],
        [[3, 1, 7, 2, 5], [9, 1, 0, 3, 2], [5, 2, 4, 8, 3], [4, 3, 1, 1, 4]],
        [[3, 1, 7, 2, 5], [5, 1, 0, 9, 2], [8, 2, 4, 9, 3], [4, 3, 1, 1, 4]]
    ]
    '''

    data = [
        [[[3, 1, 7, 2], [5, 1, 0, 9], [8, 2, 4, 9], [4, 3, 1, 1]],
         [[3, 1, 7, 2], [9, 1, 0, 3], [5, 2, 4, 8], [4, 3, 1, 1]],
         [[3, 1, 7, 2], [5, 1, 0, 9], [8, 2, 4, 9], [4, 3, 1, 1]]],

        [[[3, 1, 7, 2], [5, 1, 0, 9], [8, 2, 4, 9], [4, 3, 1, 1]],
         [[3, 1, 7, 2], [9, 1, 0, 3], [5, 2, 4, 8], [4, 3, 1, 1]],
         [[3, 1, 7, 2], [5, 1, 0, 9], [8, 2, 4, 9], [4, 3, 1, 1]]]
    ]

    data = np.asarray(data, dtype=np.float64)
    # data = np.expand_dims(data, axis=0)

    # (2, 3, 4, 4) e.g.: 2 images, with 3 channels, 4 rows (height) and 4 columns (width)
    data_shape = data.shape

    pad = 0

    new_img, pos_result = maxPool(data, filter_h=2, filter_w=2, stride=2, padding=pad)
    # new_img = np.squeeze(new_img, axis=0)
    new_img_shape = new_img.shape
    # new_img = np.squeeze(new_img, axis=0)
    new_img = np.asarray(new_img, dtype=np.float64)

    # delta_conv = np.multiply(delta_conv, dReLU(X_conv))


def max_pool_backprop_test():
    conv_data = [
        [[[3, 1, 7, 2], [5, 1, 0, 9], [8, 2, 4, 9], [4, 3, 1, 1]],
         [[3, 1, 7, 2], [9, 1, 0, 3], [5, 2, 4, 8], [4, 3, 1, 1]],
         [[3, 1, 7, 2], [5, 1, 0, 9], [8, 2, 4, 9], [4, 3, 1, 1]]]]

    conv_data = np.asarray(conv_data, dtype=np.float64)
    conv_data_shape = conv_data.shape

    maxpool_result, maxpool_pos_indices = maxPool(conv_data)
    maxpool_result_shape = maxpool_result.shape

    bp = [
        [[[1, 2], [3, 4]],
         [[5, 6], [7, 8]],
         [[9, 10], [11, 12]]]
    ]
    bp = np.asarray(bp, dtype=np.float64)
    bp_shape = bp.shape

    # 1) the gradients flowing back in the network during backprop
    #    in the network we are working with is before the flatten
    # 2) the positional indices saved during forward pass with
    #    the positions of max values
    # 3) the shape expected from the convolutional layer
    maxpool_gradients = backprop(bp, maxpool_pos_indices, conv_data.shape)
    print()


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


def backprop(gradient_values, pos_result, conv_shape):
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

            # the second part must be changed with delta_maxpool, which is the delta_0 unflattened
            # during the backpropagation
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
            if image_rectangle.shape[0] < filter_h:
                continue
            else:
                for width in range(0, image_rectangle.shape[2], stride):
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
