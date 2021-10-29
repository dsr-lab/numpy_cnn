import numpy as np
from PIL import Image

from utils import *


def fast_max_pool(inputs, stride=2, kernel_h=2, kernel_w=2, padding=0):
    # Get required variables from the input shape
    n_images, n_channels, input_h, input_w = inputs.shape

    # Transform to matrix and reshape
    input_matrix = im2col_(inputs, kernel_h, kernel_w, stride, padding)
    # Reshape in a way that allow us to have:
    # - the expected number of channels, that must be the same of the inputs
    # - the number of rows of the matrix true divided for the number of channels
    # - fill the matrix
    input_matrix = input_matrix.reshape(n_channels, input_matrix.shape[0] // n_channels, -1)
    input_matrix_shape = input_matrix.shape

    # Compute the output size
    out_h = int((input_h + 2 * padding - kernel_h) / stride) + 1
    out_w = int((input_w + 2 * padding - kernel_w) / stride) + 1

    """
    Example: input is 1 image, 3 channels and 4x4
    
    [  # number of images
        [  # number of channels
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],  # heigth and width of the input
            [[21, 22, 23, 24], [25, 26, 27, 28], [29, 30, 31, 32], [33, 34, 35, 36]],
            [[41, 42, 43, 44], [45, 46, 47, 48], [49, 50, 51, 52], [53, 54, 55, 56]],
        ]
    ]
    
    input_matrix will produce the following output
    
    1   3   9   11     
    2   4   10  12
    5   7   13  15
    6   8   14  16
    
    21  23  29  31          
    22  24  30  32
    25  27  33  35
    26  28  34  36
    
    41  43  49  51          
    42  44  50  52
    45  47  53  55
    46  48  54  56
    
    The resulting matrix is (12, 4)
    
    We have to reshape this matrix based on the 
        (
            number of channels, 
            number of input matrix rows // number of channles,
            -1
        )
    
    So, in this example the shape will be (3, 4, 4)
    
    
    
    """

    # Perform the maxpool column wise
    max_pool_result = np.max(input_matrix, axis=1)
    pos_result = np.argmax(input_matrix, axis=1)
    # Add one dimension for managing the number of the images
    max_pool_result = np.array(np.hsplit(max_pool_result, n_images))
    # Reshape to the expected shape after the max pooling operation
    max_pool_result = max_pool_result.reshape(n_images, n_channels, out_h, out_w)

    return max_pool_result, pos_result


def max_pool(input_images, stride=2, filter_h=2, filter_w=2, padding=0):
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


def fast_maxpool_backprop(gradient_values, conv_shape, padding, stride, max_pool_size, pos_result):

    n_channels = conv_shape[1]
    # values coming from gradients during the backpropagation
    # bp = [
    #     [[[1, 2], [3, 4]],
    #      [[5, 6], [7, 8]],
    #      [[9, 10], [11, 12]]],
    #     [[[13, 14], [15, 16]],
    #      [[17, 18], [19, 20]],
    #      [[21, 22], [23, 24]]]
    # ]
    # bp = np.asarray(bp, dtype=np.float64)
    bp_flattened = gradient_values.reshape(n_channels, -1)

    # the convolution shape expected is then (1,3,4,4)
    # delta_conv = np.zeros((2, 3, 4, 4))  # shape of the gradient
    delta_conv = np.zeros(conv_shape)
    delta_conv_shape = delta_conv.shape
    delta_conv_col = im2col_(delta_conv, max_pool_size, max_pool_size, stride, padding)

    # Those are indexes channel wise    n_channels = 3
    # pos_result = [[2, 3, 0, 1, 2, 3, 0, 1], [2, 0, 2, 2, 2, 0, 2, 2], [2, 0, 0, 1, 2, 0, 0, 3]]
    # pos_result = np.asarray(pos_result)
    # pos_result_shape = pos_result.shape

    row_coefficient = delta_conv_col.shape[0] // n_channels
    channels = np.arange(0, delta_conv_col.shape[0], row_coefficient)
    channels = np.repeat(channels, pos_result.shape[1])
    channels = channels.reshape(pos_result.shape)
    pos_result += channels

    col_indices = np.arange(delta_conv_col.shape[1])
    col_indices = np.tile(col_indices, n_channels)
    col_indices = col_indices.reshape(1, n_channels, -1)

    np.add.at(delta_conv_col, (pos_result, col_indices), bp_flattened)

    # delta_conv = delta_conv_col.reshape(2, 3, 4, 4)
    # delta_conv = delta_conv_col.reshape(conv_shape)

    delta_conv = col2im(delta_conv_col, conv_shape, max_pool_size, max_pool_size, stride, padding)
    # n_images = conv_shape[0]
    # delta_conv = delta_conv_col.reshape(n_images, -1)
    # delta_conv = np.array(np.hsplit(delta_conv, n_channels))
    # delta_conv = delta_conv.reshape(conv_shape)

    return delta_conv
