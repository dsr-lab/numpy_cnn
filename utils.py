import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("TkAgg")


# ################################################################################
# PLOTTING
# ################################################################################
def show_gray_scale_image(image, title=None):
    a = image.shape
    # initially (3, 32, 32), pyplot expects (32, 32, 3)
    #plt.imshow(np.transpose(image, (1, 2, 0)))
    image = image.squeeze(axis=0)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show(block=True)

    if title is not None:
        plt.title(title)


def show_image(image, title=None):
    # npimg = image.numpy()

    # initially (3, 32, 32), pyplot expects (32, 32, 3)
    plt.imshow(np.transpose(image, (1, 2, 0)))
    plt.axis('off')
    plt.show(block=True)

    if title is not None:
        plt.title(title)


# ################################################################################
# METRICS
# ################################################################################
def accuracy(scores, labels):
    n_samples = scores.shape[1]

    predictions = np.argmax(scores, axis=0)

    n_correct = (labels == predictions).sum()

    #acc = n_correct / n_samples

    acc = np.true_divide(n_correct, n_samples)

    return acc


# ################################################################################
# FAST CONVOLUTIONS AND MAX POOL UTILITY METHODS
# ################################################################################
def get_indices(input_shape, filter_h, filter_w, stride, pad):
    # Input size
    n_images, channels, image_h, image_w = input_shape

    # Output size
    out_h = int((image_h + 2 * pad - filter_h) / stride) + 1
    out_w = int((image_w + 2 * pad - filter_w) / stride) + 1

    # ##############################
    # Row indices
    # ##############################

    # Create the starting point for rows indices
    # The goal is to create an array that has:
    # - values that go from 0 to (filter_h - 1)
    # - repeat the above values (filter_w - 1) times
    #
    # Example 1:
    #   filter_h = filter_w = 2
    #   a = np.arange(filter_h) = [0, 1]
    #   np.repeat(a) = [0, 0, 1, 1]
    #
    # Example 2:
    #   filter_h = 3, filter_w = 2
    #   a = np.arange(filter_h) = [0, 1, 2]
    #   np.repeat(a) = [0, 0, 1, 1, 2, 2]
    row_indices_vector_0 = np.repeat(np.arange(filter_h), filter_w)

    # Repeat based on the number of channels
    # Example:
    #   filter_h = filter_w = 2, channels = 3
    #   a = np.arange(filter_h) = [0, 1]
    #   b = np.repeat(a) = [0, 0, 1, 1]
    #   np.tile(b, channels) = [[0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]

    row_indices_vector_0 = np.tile(row_indices_vector_0, channels)

    # Create the vector that is used for summing 1 after each level of the
    # convolution operation
    sum_vector = stride * np.repeat(np.arange(out_h), out_w)

    # At this point we need to sum rows_indices_vector_0 with the sum_vector.
    # Notice that:
    # - rows_indices_vector_0 is reshaped to a single column
    # - sum_vector is reshaped to a single row
    row_indices = row_indices_vector_0.reshape(-1, 1) + sum_vector.reshape(1, -1)

    # ##############################
    # Column indices
    # ##############################

    # As before, create the initial vector required for column indices
    # Differently from before, when we slide horizontally the filter we have
    # to increase the index a number of times equal to (filter_h-1)
    column_indices_vector_0 = np.tile(np.arange(filter_h), filter_w)
    column_indices_vector_0 = np.tile(column_indices_vector_0, channels)

    # Create the sum vector
    sum_vector = stride * np.tile(np.arange(out_h), out_w)

    # Sum the two vectors
    column_indices = column_indices_vector_0.reshape(-1, 1) + sum_vector.reshape(1, -1)

    # ----Compute matrix of index d----

    # Matrix required for considering different channels while processing
    # the row_indices and column_indices variables.
    # This is required later for reshaping to the expected output, which should not be a matrix.
    channel_matrix = np.repeat(np.arange(channels), filter_h * filter_w).reshape(-1, 1)

    return row_indices, column_indices, channel_matrix


def im2col_(images, filter_h, filter_w, stride, pad):
    # Apply the padding
    padded_images = np.pad(images, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    row_indices, col_indices, channel_matrix = get_indices(images.shape, filter_h, filter_w, stride, pad)

    # Apply the indexing to all the images
    image_matrices = padded_images[:, channel_matrix, row_indices, col_indices]

    # Create a single matrix that considers all the images concatenating along the last axis
    # (e.g.: horizontally)
    image_matrices = np.concatenate(image_matrices, axis=-1)

    return image_matrices


def col2im(dX_col, X_shape, HF, WF, stride, pad):

    a = X_shape

    # Get input size
    N, D, H, W = X_shape
    # Add padding if needed.
    H_padded, W_padded = H + 2 * pad, W + 2 * pad
    X_padded = np.zeros((N, D, H_padded, W_padded))

    # Index matrices, necessary to transform our input image into a matrix.
    i, j, d = get_indices(X_shape, HF, WF, stride, pad)
    # Retrieve batch dimension by spliting dX_col N times: (X, Y) => (N, X, Y)
    dX_col_reshaped = np.array(np.hsplit(dX_col, N))
    # Reshape our matrix back to image.
    # slice(None) is used to produce the [::] effect which means "for every elements".
    np.add.at(X_padded, (slice(None), d, i, j), dX_col_reshaped)
    X_padded_shape = X_padded.shape
    # Remove padding from new image if needed.
    if pad == 0:
        return X_padded
    elif type(pad) is int:
        # return X_padded[pad:-pad, pad:-pad, :, :]
        return X_padded[:, :, pad:-pad, pad:-pad]
