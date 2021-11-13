from utils import *


def convolve_2d(images, kernel, padding=0, stride=1):
    """
    Compute the NAIVE version of the convolution

    Parameters
    ----------
    images : ndarray
        Inputs of the convolutional layer
    kernel : ndarray
        Kernel containing the weights of the convolutional layer
    padding: int, optional
        The possible padding applied to the inputs
    stride: int, optional
        The stride applied for the convolution operation

    Returns
    -------
    convolution_result : ndarray
        The result of the computed convolution
    """

    # Get kernel shape values
    output_channels = kernel.shape[0]
    input_channels = kernel.shape[1]
    kernel_h = kernel.shape[2]
    kernel_w = kernel.shape[3]

    # Compute the expected output size (h,w)
    output_h = int(((images.shape[2] + 2 * padding - kernel.shape[2]) / stride) + 1)
    output_w = int(((images.shape[3] + 2 * padding - kernel.shape[3]) / stride) + 1)

    # Init the convolution matrix with zero values
    convolution_result = np.zeros((images.shape[0], output_channels, output_h, output_w),
                                  dtype=np.float)

    # Cycle all the images in the batch
    for image_idx in range(images.shape[0]):
        # Extract a single image
        current_image = images[image_idx, :, :, :]
        # Apply padding
        current_image = np.pad(current_image, ((0, 0), (padding, padding), (padding, padding)), mode='constant')

        # Cycle all the filters in the convolutional layer
        for filter_position in range(kernel.shape[0]):
            # Extract the current filter
            filter_selected = kernel[filter_position, :, :, :]

            # height index for the output activation
            output_h_idx = 0

            # Slide the image, starting from its height.
            #  - current_image.shape[0] = channels
            #  - current_image.shape[1] = height
            #  - current_image.shape[2] = width
            for i in range(0, current_image.shape[1], stride):
                # width index for the output activation
                output_w_idx = 0

                # Extract the sub-portion of the image
                # - get all channels
                # - height ==> from i to i + kernel_size
                # - width ==> all the width
                image_rectangle = current_image[:, i:i + kernel_h, :]
                if image_rectangle.shape[1] < kernel_h:
                    continue
                else:
                    for j in range(0, image_rectangle.shape[2], stride):
                        if j >= image_rectangle.shape[2]:
                            continue
                        image_portion = image_rectangle[:, :, j:j + kernel_w]
                        if image_portion.shape[2] < kernel_w:
                            continue
                        else:
                            # Perform the dot product
                            inner_result = np.multiply(filter_selected, image_portion)
                            convolution_result[image_idx, filter_position, output_h_idx, output_w_idx] = \
                                np.sum(inner_result)
                            output_w_idx += 1
                    output_h_idx += 1

    return convolution_result


def fast_convolve_2d(inputs, kernel, padding=0, stride=1):
    """
    Compute the FAST version of the convolution

    Parameters
    ----------
    inputs : ndarray
        Inputs of the convolutional layer
    kernel : ndarray
        Kernel containing the weights of the convolutional layer
    padding: int, optional
        The possible padding applied to the inputs
    stride: int, optional
        The stride applied for the convolution operation

    Returns
    -------
    conv_result : ndarray
        The result of the computed convolution
    """

    # Get required variables from the input shape
    n_images, n_channels, input_h, input_w = inputs.shape

    # Get required variables from the kernel shape
    out_channels, in_channels, kernel_h, kernel_w = kernel.shape

    # Compute the output size
    out_h = int((input_h + 2 * padding - kernel_h) / stride) + 1
    out_w = int((input_w + 2 * padding - kernel_w) / stride) + 1

    # Transform to matrix
    input_matrix = im2col(inputs, kernel_h, kernel_w, stride, padding)

    # Reshape the kernel based on the number of channels
    # (e.g.: one channel = one row in the resulting matrix)
    kernel_matrix = kernel.reshape((out_channels, -1))

    # perform the matrix multiplication that emulates the convolution
    conv_matrix = kernel_matrix @ input_matrix

    # reshape to the expected shape after the convolution
    conv_result = np.array(np.hsplit(conv_matrix, n_images))
    conv_result = conv_result.reshape((n_images, out_channels, out_h, out_w))

    return conv_result


def convolution_backprop(X, kernel, gradient_values, padding=0, stride=1):
    """
    Compute the NAIVE version of the backpropagation through a
    convolutional layer

    Parameters
    ----------
    X : ndarray
        Inputs of the convolutional layer
    kernel : ndarray
        Kernel containing the weights of the convolutional layer
    gradient_values : ndarray
        The gradients that are flowing back from following layers
        through the backpropagation algorithm
    padding: int, optional
        The possible padding applied to the inputs
    stride: int, optional
        The stride applied for the convolution operation

    Returns
    -------
    dW : ndarray
        The derivative computed WRT the weights.
        This basically represents a convolution between the INPUT of the convolutional
        layer, and the gradients that are flowing back from the following layer
        during the backpropagation
    dX : ndarray
        The derivative computed WRT the inputs.
        This basically represents a FULL CONVOLUTION between the FLIPPED KERNEL WEIGHTS
        of the convolutional layer, and the gradients that are flowing back from
        the following layer during the backpropagation
    """
    # Get required variables from the kernel shape
    output_channels = kernel.shape[0]
    input_channels = kernel.shape[1]
    kernel_h = kernel.shape[2]
    kernel_w = kernel.shape[3]

    # Initializing dW with the expected shape
    dW = np.zeros(kernel.shape)

    # Cycle all the inputs in the batch.
    # For simplicity, refer to inputs as images.
    for image_idx in range(X.shape[0]):
        # Extract a single image
        current_image = X[image_idx, :, :, :]
        # Apply padding
        current_image = np.pad(current_image, ((0, 0), (padding, padding), (padding, padding)), mode='constant')

        # Consider each filter in the convolutional layer independently
        for filter_position in range(output_channels):

            # Now get the portion of the image where the convolution with the gradients will be applied.

            # Slide the image, starting from its height.
            #  - current_image.shape[0] = channels
            #  - current_image.shape[1] = height
            #  - current_image.shape[2] = width
            for i in range(0, current_image.shape[1], stride):

                # Extract the first sub-part of the image
                # - get all channels
                # - height ==> from i to i + kernel_size
                # - width ==> all the widths
                image_rectangle = current_image[:, i:i + kernel_h, :]
                if image_rectangle.shape[1] < kernel_h:
                    continue
                else:
                    # Now repeat the same operation as before, but for getting a specific width,
                    # which must be equal to the kernel weight
                    for j in range(0, image_rectangle.shape[2], stride):
                        if j >= image_rectangle.shape[2]:
                            continue
                        image_portion = image_rectangle[:, :, j:j + kernel_w]
                        if image_portion.shape[2] < kernel_w:
                            continue
                        else:
                            # Multiply a specific gradient value with the portion of the input image
                            # where the convolution can be applied.
                            out = image_portion[:, :, :] * gradient_values[image_idx, filter_position, i, j]
                            # Update the derivative by summing the current values with the ones just computed.
                            # Some of the regions overlaps during the convolution operation.
                            dW[filter_position, :, :, :] += out

    # What follows is an alternative method for computing the backpropagation, using
    # directly the convolution operation. This has been commented for performance purposes.
    # The computation of dW can be seen as the convolutio between the gradient values
    # and the input image. However, it is necessary to:
    # 1) Take each single image
    # 2) From each single image, consider one channel at a time
    # 3) Consider each channel in the gradient as a single kernel
    # Finally, it is possible to compute the convolution between 2 and 3
    # dW2 = np.zeros(kernel.shape)
    # for filter_position in range(output_channels):
    #     for img_position in range(gradient_values.shape[0]):
    #         current_filter = gradient_values[img_position, filter_position, :, :]
    #         k = np.zeros((1, 1, gradient_values.shape[2], gradient_values.shape[3]))
    #         k[0, 0, :, :] = current_filter
    #
    #         img = np.zeros((1, 1, X.shape[2], X.shape[3]))
    #
    #         for ch in range(X.shape[1]):
    #             print(ch)
    #             img[0, 0, :, :] = X[img_position, ch, :, :]
    #
    #             res = convolve_2d(img, k)
    #             res = res[0, 0, :, :]
    #
    #             dW2[filter_position, ch, :, :] += res
    # sanity_check = np.isclose(dW, dW2).all()

    # Compute the gradients WRT the inputs (dX)
    # Init dX with the expected shape
    dX = np.zeros_like(X)
    # Pad if required
    dx_padded = np.pad(dX, ((0,), (0,), (padding,), (padding,)), 'constant')
    # The gradient values must be padded in order to apply a full convolution
    gradient_values_padded = np.pad(gradient_values, ((0,), (0,), (kernel_w - 1,), (kernel_h - 1,)), 'constant')

    # Flip the kernel
    kernel_flipped = np.zeros_like(kernel)
    for i in range(kernel_h):
        for j in range(kernel_w):
            kernel_flipped[:, :, i, j] = kernel[:, :, kernel_h - i - 1, kernel_w - j - 1]

    # Cycle all the images in the batch
    for n_images in range(X.shape[0]):
        # Cycle all the filters in the convolutional layer
        for filter_idx in range(output_channels):
            # Input height indices
            for input_h_idx in range(X.shape[2] + 2 * padding):
                # Input width indices
                for input_w_idx in range(X.shape[3] + 2 * padding):
                    # Kernel height indices
                    for kernel_h_idx in range(kernel_h):
                        # Kernel width indices
                        for kernel_w_idx in range(kernel_w):
                            # Cycle the channels
                            for ch in range(X.shape[1]):
                                # Now set the value of each cell in dx_padded.
                                # Some will be zeros due to the 0 padding.
                                # Multiply the padded gradient values with the flipped kernel values.
                                dx_padded[n_images, ch, input_h_idx, input_w_idx] += \
                                    gradient_values_padded[n_images, filter_idx, input_h_idx + kernel_h_idx, input_w_idx + kernel_w_idx] * kernel_flipped[
                                    filter_idx, ch, kernel_h_idx, kernel_w_idx]

    # Remove the padding from the final result if required
    dX = dx_padded[:, :, padding:dX.shape[2], padding:dX.shape[3]]

    return dW, dX


def fast_convolution_backprop(inputs, kernel, gradient_values, padding=0, stride=1):
    """
    Compute the FAST version of the backpropagation through a
    convolutional layer

    Parameters
    ----------
    inputs : ndarray
        Inputs of the convolutional layer
    kernel : ndarray
        Kernel containing the weights of the convolutional layer
    gradient_values : ndarray
        The gradients that are flowing back from following layers
        through the backpropagation algorithm
    padding: int, optional
        The possible padding applied to the inputs
    stride: int, optional
        The stride applied for the convolution operation

    Returns
    -------
    dW : ndarray
        The derivative computed WRT the weights
    dX : ndarray
        The derivative computed WRT the inputs
    """
    # Get required variables from the kernel shape
    out_channels, in_channels, kernel_h, kernel_w = kernel.shape

    # Get required variables from the inputs
    n_inputs, _, _, _ = inputs.shape

    # Transform the inputs in to plain matrices
    X_col = im2col(inputs, kernel_h, kernel_w, stride, padding)
    # Flat the kernel
    w_col = kernel.reshape((out_channels, -1))

    # Reshape dout properly.
    # (Number of images * Number of channels), (gradient height * gradient width)
    dout = gradient_values.reshape(gradient_values.shape[0] * gradient_values.shape[1],
                                   gradient_values.shape[2] * gradient_values.shape[3])

    # When working with the flat version of the gradients, then we need to stack
    # horizontally the images.
    '''
        Example:
        The gradient have the shape(n_images=2, n_channels=2, height=2, width=3)
        
        [
            [ # image 1
                [[1, 2], [3, 4]],  # channel 1 
                [[5, 6], [7, 8]]  # channel 2
            ],
            [ # image 2
                [[9, 10], [11, 12]],  # channel 1 
                [[13, 14], [15, 16]]  # channel 2
            ]
        ] 
        
        After the previus line of code we obtain:
            [
                [1, 2, 3, 4]        # img 1 - ch 1
                [5, 6, 7, 8]        # img 1 - ch 2
                [9, 10, 11, 12]     # img 2 - ch 1
                [13, 14, 15, 16]    # img 2 - ch 2
            ]
        
        Split WRT the number of images:
            [   
                [
                    [1, 2, 3, 4],        
                    [5, 6, 7, 8]
                ],        
                [
                    [9, 10, 11, 12]     
                    [13, 14, 15, 16]
                ]    
            ]
        
        Finally, stack horizontally:
                [
                    [1, 2, 3, 4, 9, 10, 11, 12],
                    [5, 6, 7, 8, 13, 14, 15, 16]
                ]
    '''
    dout = np.array(np.vsplit(dout, n_inputs))
    dout = np.concatenate(dout, axis=-1)

    # Perform matrix multiplication between reshaped dout and w_col to get dX_col.
    dX_col = w_col.T @ dout

    # Perform matrix multiplication between reshaped dout and X_col to get dW_col.
    dw_col = dout @ X_col.T

    # Reshape back to image (col2im).
    dX = col2im(dX_col, inputs.shape, kernel_h, kernel_w, stride, padding)

    # Reshape dw_col into dw.
    dW = dw_col.reshape((dw_col.shape[0], in_channels, kernel_h, kernel_w))

    return dW, dX
