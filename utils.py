import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from config import *

matplotlib.use("TkAgg")

# ################################################################################
# METRICS
# ################################################################################
def accuracy(scores, labels):
    """
    Compute the accuracy

    Parameters
    ----------
    scores : ndarray
        Scores obtained after the softmax function at the end of the network
    labels : ndarray
        Labels expected after computing the predictions

    Returns
    -------
    acc : float
        The computed accuracy
    """
    n_samples = scores.shape[0]
    predictions = np.argmax(scores, axis=1)
    n_correct = (labels == predictions).sum()

    acc = np.true_divide(n_correct, n_samples)
    return acc


# ################################################################################
# MODEL
# ################################################################################
def save_weights(weights, epoch, path):
    """
    Save weigths to file system

    Parameters
    ----------
    weights : ndarray
        The path where weights are on file system
    epoch : int
        The epoch where these weights have been obtained
    path : string
        The path where weights are on file system
    """

    # Do not save weights if running the model on a dummy dataset
    if not TRAIN_SMALL_DATASET:
        path = os.path.join(os.path.dirname(__file__), path)
        os.makedirs(path, exist_ok=True)
        np.save(f'{path}/fc1_w.npy', weights['fc1_w'])
        np.save(f'{path}/fc1_b.npy', weights['fc1_b'])
        np.save(f'{path}/fc2_w.npy', weights['fc2_w'])
        np.save(f'{path}/fc2_b.npy', weights['fc2_b'])
        np.save(f'{path}/conv1_w.npy', weights['conv1_w'])
        np.save(f'{path}/conv2_w.npy', weights['conv2_w'])
        np.save(f'{path}/epoch.npy', epoch)


def load_weights(weights_path):
    """
    Load weigths from file system

    Parameters
    ----------
    weights_path : string
        The path where weights are on file system

    Returns
    -------
    weights : ndarray
        The parameters loaded from file system
    """

    weights = {
        'conv1_w': np.load(f'{weights_path}/conv1_w.npy'),
        'conv2_w': np.load(f'{weights_path}/conv2_w.npy'),
        'fc1_w': np.load(f'{weights_path}/fc1_w.npy'),
        'fc1_b': np.load(f'{weights_path}/fc1_b.npy'),
        'fc2_w': np.load(f'{weights_path}/fc2_w.npy'),
        'fc2_b': np.load(f'{weights_path}/fc2_b.npy'),
    }
    epoch = np.load(f'{weights_path}/epoch.npy'),
    return weights, epoch


def init_optimizer_dictionary():
    """
    Initialize optimizer values

    Returns
    -------
    optimizer : ndarray
        The generated optimizer values
    """
    optimizer = {
        'momentum_w1': 0,
        'momentum_w2': 0,
        'momentum_b0': 0,
        'momentum_b1': 0,
        'momentum_conv1': 0,
        'momentum_conv2': 0,
        'velocity_w1': 0,
        'velocity_w2': 0,
        'velocity_b0': 0,
        'velocity_b1': 0,
        'velocity_conv1': 0,
        'velocity_conv2': 0
    }
    return optimizer


def init_model_weights():
    """
    Initialize model weights

    Returns
    -------
    kernel : ndarray
        The generated weigths
    """

    # Set variables according to the dataset
    # MNIST
    input_channels = 1
    fan_in = 2304
    # CIFAR10
    if USE_CIFAR_10:
        input_channels = 3
        fan_in = 3136

    if USE_HE_WEIGHT_INITIALIZATION:
        weights = {
            # Convolutional layer weights initialization
            'conv1_w': generate_kernel(input_channels=input_channels, output_channels=8, kernel_h=3, kernel_w=3),
            'conv2_w': generate_kernel(input_channels=8, output_channels=16, kernel_h=3, kernel_w=3),

            # Linear layer weights initialization
            # NOTE: using HE weight initialization (https: // arxiv.org / pdf / 1502.01852.pdf)
            'fc1_w': np.random.randn(fan_in, 64) / np.sqrt(fan_in / 2),
            'fc1_b': np.zeros((1, 64)),
            'fc2_w': np.random.randn(64, 10) / np.sqrt(64 / 2),
            'fc2_b': np.zeros((1, 10))
        }
    else:
        fc1_stdv = 1. / np.sqrt(fan_in)
        fc2_stdv = 1. / np.sqrt(64)
        weights = {
            # Convolutional layer weights initialization
            'conv1_w': generate_kernel(input_channels=input_channels, output_channels=8, kernel_h=3, kernel_w=3),
            'conv2_w': generate_kernel(input_channels=8, output_channels=16, kernel_h=3, kernel_w=3),

            # Linear layer weights initialization
            # NOTE: using HE weight initialization (https: // arxiv.org / pdf / 1502.01852.pdf)
            'fc1_w': np.random.uniform(low=-fc1_stdv, high=fc1_stdv, size=(fan_in, 64)),
            'fc1_b': np.random.uniform(low=-fc1_stdv, high=fc1_stdv, size=(1, 64)),
            'fc2_w': np.random.uniform(low=-fc1_stdv, high=fc1_stdv, size=(64, 10)),
            'fc2_b': np.random.uniform(low=-fc2_stdv, high=fc2_stdv, size=(1, 10))
        }

    return weights


def generate_kernel(input_channels=3, output_channels=16, kernel_h=3, kernel_w=3, random=True):
    """
    Generate a convolutional layer kernel

    Parameters
    ----------
    input_channels : int, optional
        The number of input channels
    output_channels : int, optional
        The number of output channels (e.g.: the number of filters in the conv layer)
    kernel_h : int, optional
        filter height
    kernel_w : int, optional
        filter width
    random: bool, optional
        Used only for debugging purposes

    Returns
    -------
    kernel : ndarray
        The generated kernel weigths
    """

    # Compute fan_in, which represets the amount of neurons of the parent layer
    receptive_field_size = kernel_h * kernel_w
    fan_in = input_channels * receptive_field_size

    if random:
        if USE_HE_WEIGHT_INITIALIZATION:
            return np.random.randn(output_channels, input_channels, kernel_h, kernel_w) / np.sqrt(fan_in / 2)
        else:
            stdv = 1. / np.sqrt(fan_in)
            return np.random.uniform(low=-stdv, high=stdv, size=(output_channels, input_channels, kernel_h, kernel_w))

    else:
        return np.ones((output_channels, input_channels, kernel_h, kernel_w)) * 2


# ################################################################################
# FAST CONVOLUTIONS AND MAX POOL UTILITY METHODS
# ################################################################################
def im2col(images, filter_h, filter_w, stride=1, padding=0):
    """
    Transform images into matrices

    Parameters
    ----------
    images : ndarray
        The array of images
    filter_h : int
        filter height
    filter_w : int
        filter width
    padding: int, optional
        The possible padding applied to the inputs
    stride: int, optional
        The stride applied for the convolution operation

    Returns
    -------
    image_matrices : ndarray
        Array containing the images in plain matrix form
    """
    # Apply the padding
    padded_images = np.pad(images, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
    row_indices, col_indices, channel_matrix = __get_indices(images.shape, filter_h, filter_w, stride, padding)

    # Get the image values at the positions indicated by the indices for creating the final matrix.
    image_matrices = padded_images[:, channel_matrix, row_indices, col_indices]

    # Create a single matrix that considers all the images concatenating along the last axis
    # (e.g.: horizontally)
    image_matrices = np.concatenate(image_matrices, axis=-1)

    return image_matrices


def col2im(dx_col, x_shape, filter_h, filter_w, stride=1, padding=0):
    """
    Transform images into matrices

    Parameters
    ----------
    dx_col : ndarray
        Derivative of the gradients WRT the input
    x_shape : ndarray
        Original input shape
    filter_h : int
        filter height
    filter_w : int
        filter width
    padding: int, optional
        The possible padding applied to the inputs
    stride: int, optional
        The stride applied for the convolution operation

    Returns
    -------
    image : ndarray
        Array containing the inputs converted back to the original shape
    """
    # Get required variables from the input shape
    n_images, n_channels, input_h, input_w = x_shape

    # Add padding if needed.
    input_h_padded, input_w_padded = input_h + 2 * padding, input_w + 2 * padding
    x_padded = np.zeros((n_images, n_channels, input_h_padded, input_w_padded))

    # Index matrices, necessary to transform our input image into a matrix.
    i, j, d = __get_indices(x_shape, filter_h, filter_w, stride, padding)

    # Retrieve batch dimension by spliting dx_col n_images times: (X, Y) => (n_images, X, Y)
    dx_col_reshaped = np.array(np.hsplit(dx_col, n_images))
    
    # Reshape our matrix back to image.
    # slice(None) is used to produce the [::] effect which means "for every elements".
    np.add.at(x_padded, (slice(None), d, i, j), dx_col_reshaped)

    # Remove padding from new image if needed.
    if padding == 0:
        return x_padded
    elif type(padding) is int:
        return x_padded[:, :, padding:-padding, padding:-padding]


def __get_indices(input_shape, filter_h, filter_w, stride=1, pad=0):
    """
    Create a matrix of indices that are required for converting
    an image (e.g.: with shape (3, 4, 4)) into a plain matrix
    (e.g.: with shape (12, 9))

    Parameters
    ----------
    input_shape : ndarray
        Shape of the input
    filter_h : int
        filter height
    filter_w : int
        filter width
    pad: int, optional
        The possible padding applied to the inputs
    stride: int, optional
        The stride applied for the convolution operation

    Returns
    -------
    row_indices : ndarray
        row positions WRT the original image
    column_indices : ndarray
        column positions WRT the original image
    channel_matrix : ndarray

    """
    # Input size
    n_images, channels, image_h, image_w = input_shape

    # Output size
    out_h = int((image_h + 2 * pad - filter_h) / stride) + 1
    out_w = int((image_w + 2 * pad - filter_w) / stride) + 1

    # ##############################
    # Row indices
    # ##############################
    '''
    MATHEMATICAL RULE FOR COMPUTING INDICES:

        N_LEVELS = out_w (e.g.: the number of time I can slide the kernel 
        vertically WRT the input)

        K = generic kernel size (e.g.: same width and height)
        N = image size

        ROW INDICES:
            LEVEL 1: 
                0, 0, ..., 1, 1, ..., k-1, k-1, ...  (where each number is repeated k times)
            LEVEL 2:
                1, 1, ..., 2, 2, ..., k, k ...
            LEVEL N-K:
                N-K, N-K, ..., N-K+1, N-K+1, ..., N-1, N-1, ...

        COL INDICES:
            LEVEL 1:
                0, 1, ..., K-1, ... (repeated K times)
            LEVEL 2:
                1, 2, ..., K, ... (repeated K times)
            LEVEL N-K:
                N-K, N-K+1, ..., N-1, ... (repeated K times)

    '''

    # Create the starting point for rows indices
    '''
    The goal is to create an array that has:
    - values that go from 0 to (filter_h - 1)
    - repeat the above values (filter_w - 1) times

    Example 1:
      filter_h = filter_w = 2
      a = np.arange(filter_h) = [0, 1]
      np.repeat(a, filter_w-1) = [0, 0, 1, 1]

    Example 2:
      filter_h = 3, filter_w = 3
      a = np.arange(filter_h) = [0, 1, 2]
      np.repeat(a, filter_w-1) = [0, 0, 1, 1, 2, 2]
    '''
    row_indices_vector_0 = np.repeat(np.arange(filter_h), filter_w)

    # Repeat based on the number of channels
    '''
    Example:
      filter_h = filter_w = 2, channels = 3
      a = np.arange(filter_h) = [0, 1]
      b = np.repeat(a) = [0, 0, 1, 1]
      np.tile(b, channels) = [[0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
    '''
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

    # Matrix required for considering different channels while processing
    # the row_indices and column_indices variables.
    # This is used for reshaping to the final expected output, which should not be a matrix.
    channel_matrix = np.repeat(np.arange(channels), filter_h * filter_w).reshape(-1, 1)

    return row_indices, column_indices, channel_matrix


# ################################################################################
# PLOTTING
# ################################################################################
def show_gray_scale_image(image, title=None):
    # initially (3, 32, 32), pyplot expects (32, 32, 3)
    #plt.imshow(np.transpose(image, (1, 2, 0)))
    image = image.squeeze(axis=0)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show(block=True)

    if title is not None:
        plt.title(title)


def show_first_layer(image, title):
    fig = plt.figure(figsize=(10, 10))
    columns = image.shape[0]
    rows = 1
    for i in range(0, columns * rows):
        fig.add_subplot(rows, columns, i+1)
        img = normalize_image_0_to_1(image[i])
        img = np.transpose(img, (1, 2, 0))
        plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.savefig(f'test/{title}.png')
    plt.close(fig)
    #plt.show(block=True)


def normalize_image_0_to_1(image):
    if image.min() < 0:
        return np.interp(image, (image.min(), image.max()), (0, 1))
    return image


def show_image(image, title=None):

    image = normalize_image_0_to_1(image)
    # npimg = image.numpy()

    # initially (3, 32, 32), pyplot expects (32, 32, 3)
    plt.imshow(np.transpose(image, (1, 2, 0)))
    plt.axis('off')
    plt.show(block=True)

    if title is not None:
        plt.title(title)