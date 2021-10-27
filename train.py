from cifar10 import Cifar10
from flatten import flatten
from relu import ReLU, dReLU
from sanity_checks import *
from softmax import *
from utils import *
from cross_entropy import *

BATCH_SIZE = 128
EPOCHS = 80


def train_network(train_images, train_labels,
                  test_images, test_labels,
                  valid_images, valid_labels,
                  use_fast_conv):
    # TODO: optimize this without creating a copy of the dataset
    train_images_batches = np.split(train_images, np.arange(BATCH_SIZE, len(train_images), BATCH_SIZE))
    train_images_labels = np.split(train_labels, np.arange(BATCH_SIZE, len(train_labels), BATCH_SIZE))

    val_images_batches = np.split(valid_images, np.arange(BATCH_SIZE, len(valid_images), BATCH_SIZE))
    val_images_labels = np.split(valid_labels, np.arange(BATCH_SIZE, len(valid_labels), BATCH_SIZE))

    # Kernel for the convolution layer
    kernel = init_random_kernel()
    # kernel = [[[[1, 3], [5, -4]]]]
    # kernel = np.asarray(kernel, dtype=np.float64)
    # a = kernel.shape

    # fc1_w = np.random.rand(60, 450) / np.sqrt(450)
    # fc1_b = np.zeros((60, 1)) / np.sqrt(450)
    fc1_stdv = 1. / np.sqrt(3136)
    fc1_w = np.random.uniform(low=-fc1_stdv, high=fc1_stdv, size=(60, 3136))
    fc1_b = np.random.uniform(low=-fc1_stdv, high=fc1_stdv, size=(60, 1))

    # fc2_w = np.random.rand(10, 60) / np.sqrt(60)
    # fc2_b = np.zeros((10, 1)) / np.sqrt(60)
    fc2_stdv = 1. / np.sqrt(60)
    fc2_w = np.random.uniform(low=-fc2_stdv, high=fc2_stdv, size=(10, 60))
    fc2_b = np.random.uniform(low=-fc2_stdv, high=fc2_stdv, size=(10, 1))

    learning_rate = 1e-3

    beta1 = 0.9
    beta2 = 0.995
    momentum_w1 = 0
    momentum_w2 = 0
    momentum_b0 = 0
    momentum_b1 = 0
    momentum_conv1 = 0
    velocity_w1 = 0
    velocity_w2 = 0
    velocity_b0 = 0
    velocity_b1 = 0
    velocity_conv1 = 0

    for e in range(EPOCHS):

        batch_loss = 0
        batch_acc = 0
        train_samples = 0

        for idx, input_data in enumerate(train_images_batches):
            train_samples += input_data.shape[0]

            input_labels = train_images_labels[idx]

            # input_data = np.asarray([train_images[0], images[1]])
            # input_labels = np.asarray([labels[0], labels[1]])

            # One hot encoding
            one_hot_encoding_labels = np.zeros((input_labels.size, 10))
            for i in range(input_labels.shape[0]):
                position = int(input_labels[i])
                one_hot_encoding_labels[i, position] = 1
            one_hot_encoding_labels = one_hot_encoding_labels.T

            # ####################
            # Forward Pass
            # ####################
            if use_fast_conv:
                x_conv = fast_convolve_2d(input_data, kernel)
            else:
                x_conv = convolve_2d(input_data, kernel)

            conv_out_shape = x_conv.shape
            x = ReLU(x_conv)
            x_maxpool, pos_maxpool_pos = max_pool(x)
            a = x_maxpool.shape
            x_flatten = flatten(x_maxpool)

            # First fc layer
            fc1 = np.matmul(fc1_w, x_flatten) + fc1_b
            fc2_input = ReLU(fc1)

            # Second fc layer
            fc2 = np.matmul(fc2_w, fc2_input) + fc2_b

            # Finally apply the softmax
            scores = softmax(fc2)

            # Compute the cross entropy loss
            ce = cross_entropy(scores, one_hot_encoding_labels) * input_data.shape[0]

            # Compute prediction and accuracy
            acc = accuracy(scores, input_labels) * input_data.shape[0]

            # compute for the entire epoch!
            batch_acc += acc
            batch_loss += ce

            # ####################
            # Backward Pass
            # ####################

            # Start computing the derivatives required from the backpropagation algorithm
            # The 1st derivative is the one related to the softmax.
            # The softmax is a vector, therefore we have to compute the Jacobian.
            # In each cell of the Jacobian we have the partial derivative of the i-th output WRT the j-th input.

            # The input of the softmax is the fc2_output
            # The output of the softmax is a vector, whose element sum up to one.

            delta_2 = (scores - one_hot_encoding_labels)
            d_fc2_w = delta_2 @ fc2_input.T
            d_fc2_b = np.sum(delta_2, axis=1, keepdims=True)

            delta_1 = np.multiply(fc2_w.T @ delta_2, dReLU(fc1))
            d_fc1_w = delta_1 @ x_flatten.T
            d_fc1_b = np.sum(delta_1, axis=1, keepdims=True)

            # gradient WRT x0
            delta_0 = np.multiply(fc1_w.T @ delta_1, 1.0)

            # unflatten operation
            delta_maxpool = delta_0.reshape(x_maxpool.shape)

            # gradients through the maxpool operation
            delta_conv = maxpool_backprop(delta_maxpool, pos_maxpool_pos, conv_out_shape)
            delta_conv = np.multiply(delta_conv, dReLU(x_conv))

            if use_fast_conv:
                conv1_delta = fast_convolution_backprop(input_data, kernel, delta_conv)
            else:
                conv1_delta = convolution_backprop(input_data, kernel, delta_conv)


            # conv2_delta = test_conv_back(input_data, kernel, delta_conv)

            momentum_w1 = beta1 * momentum_w1 + ((1 - beta1) * d_fc1_w)
            momentum_w2 = beta1 * momentum_w2 + ((1 - beta1) * d_fc2_w)
            momentum_b0 = beta1 * momentum_b0 + ((1 - beta1) * d_fc1_b)
            momentum_b1 = beta1 * momentum_b1 + ((1 - beta1) * d_fc2_b)
            momentum_conv1 = beta1 * momentum_conv1 + ((1 - beta1) * conv1_delta)
            velocity_w1 = beta2 * velocity_w1 + ((1 - beta2) * d_fc1_w ** 2)
            velocity_w2 = beta2 * velocity_w2 + ((1 - beta2) * d_fc2_w ** 2)
            velocity_b0 = beta2 * velocity_b0 + ((1 - beta2) * d_fc1_b ** 2)
            velocity_b1 = beta2 * velocity_b1 + ((1 - beta2) * d_fc2_b ** 2)
            velocity_conv1 = beta2 * velocity_conv1 + ((1 - beta2) * conv1_delta ** 2)

            kernel = kernel - learning_rate * momentum_conv1 / np.sqrt(velocity_conv1 + 0.0000001)
            fc1_w = fc1_w - learning_rate * momentum_w1 / np.sqrt(velocity_w1 + 0.0000001)
            fc2_w = fc2_w - learning_rate * momentum_w2 / np.sqrt(velocity_w2 + 0.0000001)
            fc1_b = fc1_b - learning_rate * momentum_b0 / np.sqrt(velocity_b0 + 0.0000001)
            fc2_b = fc2_b - learning_rate * momentum_b1 / np.sqrt(velocity_b1 + 0.0000001)

            # fc2_w = fc2_w - learning_rate * d_fc2_w
            # fc2_b = fc2_b - learning_rate * d_fc2_b
            #
            # fc1_w = fc1_w - learning_rate * d_fc1_w
            # fc1_b = fc1_b - learning_rate * d_fc1_b
            #
            # kernel = kernel - learning_rate * conv1_delta

        # ##############################
        # VALIDATION
        # ##############################
        valid_batch_loss = 0
        valid_batch_acc = 0
        valid_samples = 0
        valid_batch_torch_loss = 0
        valid_batch_torch_acc = 0
        for idx, input_data in enumerate(val_images_batches):
            valid_samples += input_data.shape[0]

            input_labels = val_images_labels[idx]

            # One hot encoding
            one_hot_encoding_labels = np.zeros((input_labels.size, 10))
            for i in range(input_labels.shape[0]):
                position = int(input_labels[i])
                one_hot_encoding_labels[i, position] = 1
            one_hot_encoding_labels = one_hot_encoding_labels.T

            # ####################
            # Forward Pass
            # ####################
            if use_fast_conv:
                x_conv = fast_convolve_2d(input_data, kernel)
            else:
                x_conv = convolve_2d(input_data, kernel)
            conv_out_shape = x_conv.shape
            x = ReLU(x_conv)
            x_maxpool, pos_maxpool_pos = max_pool(x)
            a = x_maxpool.shape
            x_flatten = flatten(x_maxpool)

            # First fc layer
            fc1 = np.matmul(fc1_w, x_flatten) + fc1_b
            fc2_input = ReLU(fc1)

            # Second fc layer
            fc2 = np.matmul(fc2_w, fc2_input) + fc2_b

            # Finally apply the softmax
            scores = softmax(fc2)

            # Compute the cross entropy loss
            ce = cross_entropy(scores, one_hot_encoding_labels) * input_data.shape[0]

            # Compute prediction and accuracy
            acc = accuracy(scores, input_labels) * input_data.shape[0]

            # compute for the entire epoch!
            valid_batch_acc += acc
            valid_batch_loss += ce

        print(train_samples)
        print('Epoch: {} - Accuracy: {} - Loss: {}'.
              format(e, (batch_acc / train_samples) / 1, (batch_loss / train_samples)))
        print(valid_samples)
        print('Epoch: {} - valid_batch_acc: {} - valid_batch_loss: {}'.
              format(e, valid_batch_acc / valid_samples, valid_batch_loss / valid_samples))
        print('***********************************')


def main():
    dataset = Cifar10()

    train_images, train_labels, \
    validation_images, validation_labels, \
    test_images, test_labels = dataset.get_small_datasets()

    train_network(train_images, train_labels, validation_images, validation_labels, test_images, test_labels, True)

    #convolution_method_comparisons()
    # test_naive_fast_max_pool()
    #max_pool_backprop_test()



if __name__ == '__main__':
    main()
