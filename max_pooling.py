import numpy as np
from PIL import Image

from utils import *


def test_max_pool():

    data = [
        [[3, 1, 7, 2, 5], [5, 1, 0, 9, 2], [8, 2, 4, 9, 3], [4, 3, 1, 1, 4]],
        [[3, 1, 7, 2, 5], [9, 1, 0, 3, 2], [5, 2, 4, 8, 3], [4, 3, 1, 1, 4]],
        [[3, 1, 7, 2, 5], [5, 1, 0, 9, 2], [8, 2, 4, 9, 3], [4, 3, 1, 1, 4]]
    ]
    data = np.asarray(data, dtype=np.float32)
    data = np.expand_dims(data, axis=0)
    pad = 0

    new_img = maxPool(data, filter_h=2, filter_w=2, stride=2, padding=pad)
    # new_img = np.squeeze(new_img, axis=0)
    new_img = np.squeeze(new_img, axis=0)
    new_img = np.asarray(new_img, dtype=np.float32)

    print(new_img.shape)


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

    # Cycle all the images in the batch
    for i in range(maxpool_result.shape[0]):
        current_image = input_images[i, :, :, :]
        current_image = np.pad(current_image, ((0, 0), (padding, padding), (padding, padding)), mode='constant')
        single_maxpool_result = \
            __process_single_image(current_image, stride, output_h, output_w, filter_h, filter_w)
        maxpool_result[i, :, :, :] = single_maxpool_result

    return maxpool_result


def __process_single_image(image, stride, output_h, output_w, filter_h, filter_w):
    print('image.shape: {}'.format(image.shape))

    # Init the maxpool matrix result with zero values
    maxpool_result = np.zeros((
        image.shape[0],
        output_h,
        output_w
    ))

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
                for width in range(0, image_rectangle.shape[1], stride):
                    image_portion = image_rectangle[:, width:width + filter_w]
                    if image_portion.shape[1] < filter_w:
                        continue
                    else:
                        # Perform the max pooling
                        maxpool_result[channel, output_h_idx, output_w_idx] = \
                            np.max(image_portion)
                        output_w_idx += 1
                output_h_idx += 1

    return maxpool_result
