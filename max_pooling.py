from utils import *


def max_pool(input_images, stride=2, filter_h=2, filter_w=2, padding=0):
    """
    Compute the NAIVE version of the maxpooling operation

    Parameters
    ----------
    input_images : ndarray
        Inputs of the layer
    filter_h : int, optional
        The height of the kernel
    filter_w : int, optional
        The width of the kernel
    padding: int, optional
        The possible padding applied to the inputs
    stride: int, optional
        The stride applied

    Returns
    -------
    max_pool_result : ndarray
        The result of the computed maxpooling operation
    pos_result : ndarray
        The indices where the maxpooling operation has been applied
    """

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
    """
    Method that performs the convolution operation on a single image.
    Used only with the NAIVE version of the maxpooling operation.

    Parameters
    ----------
    image : ndarray
        Inputs of the layer
    output_h: int
        Expected output height
    output_w: int
        Expected output width
    filter_h : int, optional
        The height of the kernel
    filter_w : int, optional
        The width of the kernel
    stride: int, optional
        The stride applied

    Returns
    -------
    maxpool_result : ndarray
        The result of the computed maxpooling operation
    pos_vector : ndarray
        The indices where the maxpooling operation has been applied
    """

    # Init the maxpool matrix result with zero values
    maxpool_result = np.zeros((
        image.shape[0],
        output_h,
        output_w
    ))

    pos_vector = []

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
                        # get the indexes where the maximum value has been found:
                        # - the argmax without axis returns the index of the maximum element of the flattened array
                        # - the unravel_index extract the row and the column by considering the index
                        #   of the flattened array explained above
                        row, column = np.unravel_index(image_portion.argmax(), image_portion.shape)

                        '''
                        Pos vector detail:
                        1) original image channel
                        2) original image row
                        3) original image column
                        5) maxpooled row
                        6) maxpooled column
                        '''
                        pos_vector.append([channel, row + height, column + width, output_h_idx, output_w_idx])

                        # Perform the max pooling
                        maxpool_result[channel, output_h_idx, output_w_idx] = \
                            np.max(image_portion)
                        output_w_idx += 1
                output_h_idx += 1

    return maxpool_result, pos_vector


def fast_max_pool(inputs, stride=2, kernel_h=2, kernel_w=2, padding=0):
    """
    Compute the FAST version of the maxpooling operation

    Parameters
    ----------
    inputs : ndarray
        Inputs of the layer
    kernel_h : int, optional
        The height of the kernel
    kernel_w : int, optional
        The width of the kernel
    padding: int, optional
        The possible padding applied to the inputs
    stride: int, optional
        The stride applied

    Returns
    -------
    max_pool_result : ndarray
        The result of the computed maxpooling operation
    pos_result : ndarray
        The indices where the maxpooling operation has been applied
    """

    # Get required variables from the input shape
    n_images, n_channels, input_h, input_w = inputs.shape

    # Transform to matrix and reshape
    input_matrix = im2col(inputs, kernel_h, kernel_w, stride, padding)

    # Reshape in a way that allow us to have:
    # - the expected number of channels, that must be the same of the inputs
    # - the number of rows of the matrix true divided for the number of channels
    # - fill the matrix
    input_matrix = input_matrix.reshape(n_channels, input_matrix.shape[0] // n_channels, -1)

    # Compute the output sizes
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
    
    So, in this example the shape will be (3, 4, 4), which is the expected output
    after the maxpooling
    """

    # Perform the maxpool column wise
    max_pool_result = np.max(input_matrix, axis=1)

    # Get the indices where the maximum values have been found
    # NOTE: this is required during the backpropagation)
    pos_result = np.argmax(input_matrix, axis=1)

    # Add one dimension for managing the number of the images
    max_pool_result = np.array(np.hsplit(max_pool_result, n_images))

    # Reshape to the expected shape after the max pooling operation
    max_pool_result = max_pool_result.reshape(n_images, n_channels, out_h, out_w)

    return max_pool_result, pos_result


def maxpool_backprop(gradient_values, pos_result, conv_shape):
    """
    Compute the NAIVE backpropagation version through maxpooling layer

    Parameters
    ----------
    gradient_values : ndarray
        Gradient coming from the following layer in the network
    pos_result : int, optional
        The position where the maxpooling was applied during the forward pass
    conv_shape : int, optional
        The expected output shape


    Returns
    -------
    delta_conv : ndarray
        The result of the backpropagation operation
    """

    delta_conv = np.zeros(conv_shape)

    for image in range(len(pos_result)):
        indices = pos_result[image]
        for p in indices:
            '''
                p contains the following values:
                0) original image channel
                1) original image row
                2) original image column
                3) maxpooled row
                4) maxpooled column
            '''

            delta_conv[image, p[0], p[1], p[2]] = gradient_values[image, p[0], p[3], p[4]]
    return delta_conv


def fast_maxpool_backprop(gradient_values, conv_shape, pos_result, padding=0, stride=2, max_pool_size=2):
    """
    Compute the NAIVE backpropagation version through maxpooling layer

    Parameters
    ----------
    gradient_values : ndarray
        Gradient coming from the following layer in the network
    pos_result : int, optional
        The position where the maxpooling was applied during the forward pass
    conv_shape : int, optional
        The expected output shape
    padding: int, optional
        The possible padding applied to the inputs
    stride: int, optional
        The stride applied
    max_pool_size : int, optional
        The kernel size

    Returns
    -------
    delta_conv : ndarray
        The result of the backpropagation operation
    """
    n_channels = conv_shape[1]

    bp_flattened = gradient_values.reshape(gradient_values.shape[0] * gradient_values.shape[1],
                                           gradient_values.shape[2] * gradient_values.shape[3])
    bp_flattened = np.array(np.vsplit(bp_flattened, conv_shape[0]))
    bp_flattened = np.concatenate(bp_flattened, axis=-1)

    delta_conv = np.zeros(conv_shape)
    delta_conv_col = im2col(delta_conv, max_pool_size, max_pool_size, stride, padding)

    row_coefficient = delta_conv_col.shape[0] // n_channels
    channels = np.arange(0, delta_conv_col.shape[0], row_coefficient)
    channels = np.repeat(channels, pos_result.shape[1])
    channels = channels.reshape(pos_result.shape)
    pos_result += channels

    col_indices = np.arange(delta_conv_col.shape[1])
    col_indices = np.tile(col_indices, n_channels)
    col_indices = col_indices.reshape(1, n_channels, -1)

    np.add.at(delta_conv_col, (pos_result, col_indices), bp_flattened)

    delta_conv = col2im(delta_conv_col, conv_shape, max_pool_size, max_pool_size, stride, padding)

    return delta_conv
