from cifar10 import Cifar10
from dropout import *
from flatten import flatten
from relu import ReLU, dReLU
from sanity_checks import *
from softmax import *
from utils import *
from cross_entropy import *
from timeit import default_timer as timer

BATCH_SIZE = 128
EPOCHS = 35
CONV_DROPOUT_PROBABILITY = 0.8
DENSE_DROPOUT_PROBABILITY = 0.8

OPTIMIZER = "ADAM"  # Valid values: ADAM, MOMENTUM
CONV_PADDING = 1
TRAIN_SMALL_DATASET = False


def train_network(train_images, train_labels,
                  test_images, test_labels,
                  valid_images, valid_labels,
                  use_fast_conv,
                  use_dropout):
    # TODO: optimize this without creating a copy of the dataset
    train_images_batches = np.split(train_images, np.arange(BATCH_SIZE, len(train_images), BATCH_SIZE))
    train_images_labels = np.split(train_labels, np.arange(BATCH_SIZE, len(train_labels), BATCH_SIZE))

    val_images_batches = np.split(valid_images, np.arange(BATCH_SIZE, len(valid_images), BATCH_SIZE))
    val_images_labels = np.split(valid_labels, np.arange(BATCH_SIZE, len(valid_labels), BATCH_SIZE))

    # Kernel for the convolution layer
    kernel = generate_kernel(input_channels=3, output_channels=32, kernel_h=3, kernel_w=3)

    fc1_stdv = 1. / np.sqrt(8192)
    fc1_w = np.random.uniform(low=-fc1_stdv, high=fc1_stdv, size=(128, 8192))  # 8192 is the size after the maxpool
    fc1_b = np.random.uniform(low=-fc1_stdv, high=fc1_stdv, size=(128, 1))

    fc2_stdv = 1. / np.sqrt(128)
    fc2_w = np.random.uniform(low=-fc2_stdv, high=fc2_stdv, size=(10, 128))
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

        train_batch_loss = 0
        train_batch_acc = 0
        train_samples = 0
        start = timer()

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

            # ################################################################################
            # FORWARD PASS
            # ################################################################################
            if use_fast_conv:
                x_conv = fast_convolve_2d(input_data, kernel, padding=CONV_PADDING)
            else:
                x_conv = convolve_2d(input_data, kernel, padding=CONV_PADDING)

            # Save the shape for later use (used during the backpropagation)
            conv_out_shape = x_conv.shape

            #if use_dropout:
            #    x_conv = cnn_dropout(x_conv, CONV_DROPOUT_PROBABILITY)

            x = ReLU(x_conv)
            if use_fast_conv:
                x_maxpool, pos_maxpool_pos = fast_max_pool(x)
            else:
                x_maxpool, pos_maxpool_pos = max_pool(x)

            x_flatten = flatten(x_maxpool)

            # First fc layer
            fc1 = np.matmul(fc1_w, x_flatten) + fc1_b
            fc2_input = ReLU(fc1)
            a = fc2_input.shape

            if use_dropout:
                fc2_input = dense_dropout(fc2_input, DENSE_DROPOUT_PROBABILITY)

            # Second fc layer
            fc2 = np.matmul(fc2_w, fc2_input) + fc2_b

            # Apply the softmax for computing the scores
            scores = softmax(fc2)

            # Compute the cross entropy loss
            ce = cross_entropy(scores, one_hot_encoding_labels) * input_data.shape[0]

            # Compute prediction and accuracy
            acc = accuracy(scores, input_labels) * input_data.shape[0]

            # compute for the entire epoch!
            train_batch_acc += acc
            train_batch_loss += ce

            # ################################################################################
            # BACKWARD PASS
            # ################################################################################
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
            if use_fast_conv:
                delta_conv = fast_maxpool_backprop(
                    delta_maxpool,
                    conv_out_shape,
                    padding=0,  # not working with padding, seems to be related to conv padding
                    stride=2,
                    max_pool_size=2,
                    pos_result=pos_maxpool_pos)
            else:
                delta_conv = maxpool_backprop(delta_maxpool, pos_maxpool_pos, conv_out_shape)

            delta_conv = np.multiply(delta_conv, dReLU(x_conv))

            if use_fast_conv:
                conv1_delta = fast_convolution_backprop(input_data, kernel, delta_conv, padding=CONV_PADDING)
            else:
                conv1_delta = convolution_backprop(input_data, kernel, delta_conv, padding=CONV_PADDING)

            if OPTIMIZER == 'ADAM':
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

            elif OPTIMIZER == 'MOMENTUM':
                velocity_w1 = beta1 * velocity_w1 + ((1 - beta1) * d_fc1_w)
                velocity_w2 = beta1 * velocity_w2 + ((1 - beta1) * d_fc2_w)
                velocity_b0 = beta1 * velocity_b0 + ((1 - beta1) * d_fc1_b)
                velocity_b1 = beta1 * velocity_b1 + ((1 - beta1) * d_fc2_b)
                velocity_conv1 = beta1 * velocity_conv1 + ((1 - beta1) * conv1_delta)

                kernel = kernel - learning_rate * velocity_conv1
                fc1_w = fc1_w - learning_rate * velocity_w1
                fc2_w = fc2_w - learning_rate * velocity_w2
                fc1_b = fc1_b - learning_rate * velocity_b0
                fc2_b = fc2_b - learning_rate * velocity_b1

            else:
                fc2_w = fc2_w - learning_rate * d_fc2_w
                fc2_b = fc2_b - learning_rate * d_fc2_b

                fc1_w = fc1_w - learning_rate * d_fc1_w
                fc1_b = fc1_b - learning_rate * d_fc1_b

                kernel = kernel - learning_rate * conv1_delta

        # ################################################################################
        # VALIDATION
        # ################################################################################
        valid_batch_loss = 0
        valid_batch_acc = 0
        valid_samples = 0
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
                x_conv = fast_convolve_2d(input_data, kernel, padding=CONV_PADDING)
            else:
                x_conv = convolve_2d(input_data, kernel, padding=CONV_PADDING)

            x = ReLU(x_conv)

            if use_fast_conv:
                x_maxpool, pos_maxpool_pos = fast_max_pool(x)
            else:
                x_maxpool, pos_maxpool_pos = max_pool(x)

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

        end = timer()

        print('=== EPOCH: {} ==='.format(e))
        print('TRAIN Accuracy: {:.3f}\tTRAIN Loss: {:.3f}'.
              format(train_batch_acc / train_samples, train_batch_loss / train_samples))

        print('VALID Accuracy: {:.3f}\t\tVALID Loss: {:.3f}'.
              format(valid_batch_acc / valid_samples, valid_batch_loss / valid_samples))
        print("Elapsed time (s): {}".format(end - start))
        print()


def main():
    dataset = Cifar10()

    train_images, train_labels, \
        validation_images, validation_labels, \
        test_images, test_labels = dataset.get_small_datasets() if TRAIN_SMALL_DATASET else dataset.get_datasets()

    train_network(train_images, train_labels,
                  validation_images, validation_labels,
                  test_images, test_labels,
                  True, True)

    print()


if __name__ == '__main__':
    main()
