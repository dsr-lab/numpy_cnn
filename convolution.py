import numpy as np
from PIL import Image

from utils import show_image, show_gray_scale_image


def test_convolution():
    img = Image.open("tests/sample.jpeg")
    data = np.transpose(img, (2, 0, 1))
    data = np.asarray(data, dtype="int32")

    show_image(data)

    # The convolution layer could process more than one image per time
    # depending on the batch size
    data = np.expand_dims(data, axis=0)

    # X = np.array([[1, 0, 0], [1, 2, 3], [3, 4, 5], [7, 7, 7], [8, 8, 8], [9, 9, 9]])
    # X = X.reshape(2, 1, 3, 3)

    # conv1 = np.array([[1, 0], [0, 1]])
    # conv1 = conv1.reshape(1, 1, 2, 2)

    edge_detection_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    # kernel should be flipped. Actually not relevant for NN
    edge_detection_kernel = np.flipud(np.fliplr(edge_detection_kernel))
    edge_detection_kernel = edge_detection_kernel.reshape((1, 1, 3, 3))

    res = convolve_2d(data, edge_detection_kernel, padding=2)
    res = np.squeeze(res, axis=0)
    res = np.squeeze(res, axis=0)

    show_gray_scale_image(res)


# numberOfFilters, out_channels, kernel_size, stride
def convolve_2d(images, kernel, padding=0, stride=1):

    output_channels = kernel.shape[0]
    input_channels = kernel.shape[1]
    kernel_h = kernel.shape[2]
    kernel_w = kernel.shape[3]

    # Compute the expected output size (h,w)
    output_h = int((images.shape[2] + 2 * padding - kernel.shape[2]) + 1)
    output_w = int((images.shape[3] + 2 * padding - kernel.shape[3]) + 1)

    # Init the convolution matrix with random values
    # convolution_result = np.random.rand(images.shape[0], output_channels, output_h, output_w)
    # a = convolution_result.shape

    # Init the convolution matrix with zero values
    convolution_result = np.zeros((images.shape[0], output_channels, output_h, output_w))
    # Cycle all the images in the batch
    for image_idx in range(images.shape[0]):
        # Extract a single image
        current_image = images[image_idx, :, :, :]

        # Cycle all the filters in the convolutional layer
        for filter_position in range(kernel.shape[0]):
            # Extract the current filter
            filter_selected = kernel[filter_position, :, :, :]

            # Slide the image, starting from its height
            for i in range(0, current_image.shape[1], stride):
                # Extract the sub-portion of the image
                # - get all channels
                # - height ==> from i to i + kernel_size
                # - width ==> all the width
                image_rectangle = current_image[:, i:i + kernel_h, :]
                if image_rectangle.shape[1] < kernel_h:
                    continue
                else:
                    for j in range(0, image_rectangle.shape[2], stride):
                        image_portion = image_rectangle[:, :, j:j + kernel_w]
                        if image_portion.shape[2] < kernel_w:
                            continue
                        else:
                            inner_result = np.multiply(filter_selected, image_portion)
                            convolution_result[image_idx, filter_position, i, j] = np.sum(inner_result)

    return convolution_result
