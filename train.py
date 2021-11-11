from cifar10 import Cifar10
from dropout import *
from flatten import flatten
from mnist import Mnist
from relu import ReLU, dReLU
from sanity_checks import *
from softmax import *
from utils import *
from cross_entropy import *
from timeit import default_timer as timer
from config import *


def forward(input_data, input_labels, input_labels_one_hot_encoded, weights):
    n_samples = input_data.shape[0]

    # ********************
    # CONV 1 + RELU
    # ********************
    if USE_FAST_CONV:
        conv1_output = fast_convolve_2d(input_data, weights['conv1_w'], padding=CONV_PADDING)
    else:
        conv1_output = convolve_2d(input_data, weights['conv1_w'], padding=CONV_PADDING)

    if USE_DROPOUT:
        conv1_output = cnn_dropout(conv1_output, CONV_DROPOUT_PROBABILITY)

    conv2_input = ReLU(conv1_output)

    # ********************
    # CONV 2 + RELU
    # ********************
    if USE_FAST_CONV:
        conv2_output = fast_convolve_2d(conv2_input, weights['conv2_w'], padding=CONV_PADDING)
    else:
        conv2_output = convolve_2d(conv2_input, weights['conv2_w'], padding=CONV_PADDING)

    if USE_DROPOUT:
        conv2_output = cnn_dropout(conv2_output, CONV_DROPOUT_PROBABILITY)

    maxpool_input = ReLU(conv2_output)

    # ********************
    # MAXPOOL
    # ********************
    if USE_FAST_CONV:
        x_maxpool_output, pos_maxpool_pos = fast_max_pool(maxpool_input)
    else:
        x_maxpool_output, pos_maxpool_pos = max_pool(maxpool_input)

    # ********************
    # FLATTEN + FCs
    # ********************
    fc1_input = flatten(x_maxpool_output)

    # First fc layer
    fc1_output = np.matmul(fc1_input, weights['fc1_w']) + weights['fc1_b']
    fc2_input = ReLU(fc1_output)

    if USE_DROPOUT:
        fc2_input = dense_dropout(fc2_input, DENSE_DROPOUT_PROBABILITY)

    # ********************
    # SCORE AND LOSS
    # ********************
    # Second fc layer
    fc2_output = np.matmul(fc2_input, weights['fc2_w']) + weights['fc2_b']

    # Apply the softmax for computing the scores
    scores = softmax(fc2_output)

    # Compute the cross entropy loss
    loss = cross_entropy(scores, input_labels_one_hot_encoded) * n_samples

    # Compute prediction and accuracy
    acc = accuracy(scores, input_labels) * n_samples

    # Values required during backpropagation
    cache = {
        'fc2_input': fc2_input,
        'fc1_input': fc1_input,
        'fc1_output': fc1_output,
        'x_maxpool_output': x_maxpool_output,
        'pos_maxpool_pos': pos_maxpool_pos,
        'conv2_output': conv2_output,
        'conv2_input': conv2_input,
        'conv1_output': conv1_output
    }

    return scores, cache, loss, acc


def backward(input_data, input_labels_one_hot_encoded, scores, cache, weights, optimizer, n_weight_updates):
    # https://stats.stackexchange.com/questions/183840/sum-or-average-of-gradients-in-mini-batch-gradient-decent/183990
    delta_2 = (scores - input_labels_one_hot_encoded) / BATCH_SIZE
    d_fc2_w = cache['fc2_input'].T @ delta_2
    d_fc2_b = np.sum(delta_2, axis=0, keepdims=True)

    delta_1 = np.multiply((weights['fc2_w'] @ delta_2.T).T, dReLU(cache['fc1_output']))
    d_fc1_w = cache['fc1_input'].T @ delta_1
    d_fc1_b = np.sum(delta_1, axis=0, keepdims=True)

    # gradient WRT x0
    delta_0 = (weights['fc1_w'] @ delta_1.T).T

    # unflatten operation
    delta_0_unflattened = delta_0.reshape(cache['x_maxpool_output'].shape)

    # gradients through the maxpool operation
    if USE_FAST_CONV:
        delta_maxpool = fast_maxpool_backprop(delta_0_unflattened, cache['conv2_output'].shape,
                                              cache['pos_maxpool_pos'])
    else:
        delta_maxpool = maxpool_backprop(delta_0_unflattened, cache['pos_maxpool_pos'], cache['conv2_output'].shape)

    delta_maxpool_w = np.multiply(delta_maxpool, dReLU(cache['conv2_output']))

    if USE_FAST_CONV:
        conv2_delta_w, conv2_delta_x = \
            fast_convolution_backprop(cache['conv2_input'], weights['conv2_w'], delta_maxpool_w, padding=CONV_PADDING)
    else:
        conv2_delta_w, conv2_delta_x = \
            convolution_backprop(cache['conv2_input'], weights['conv2_w'], delta_maxpool_w, padding=CONV_PADDING)

    conv2_delta_x = np.multiply(conv2_delta_x, dReLU(cache['conv1_output']))

    if USE_FAST_CONV:
        conv1_delta_w, _ = fast_convolution_backprop(input_data, weights['conv1_w'], conv2_delta_x,
                                                     padding=CONV_PADDING)
    else:
        conv1_delta_w, _ = convolution_backprop(input_data, weights['conv1_w'], conv2_delta_x, padding=CONV_PADDING)

    # ********************
    # WEIGHT UPDATES
    # ********************
    if OPTIMIZER == 'ADAM':
        optimizer['momentum_w1'] = BETA1 * optimizer['momentum_w1'] + ((1 - BETA1) * d_fc1_w)
        optimizer['momentum_w2'] = BETA1 * optimizer['momentum_w2'] + ((1 - BETA1) * d_fc2_w)
        optimizer['momentum_b0'] = BETA1 * optimizer['momentum_b0'] + ((1 - BETA1) * d_fc1_b)
        optimizer['momentum_b1'] = BETA1 * optimizer['momentum_b1'] + ((1 - BETA1) * d_fc2_b)
        optimizer['momentum_conv1'] = BETA1 * optimizer['momentum_conv1'] + ((1 - BETA1) * conv1_delta_w)
        optimizer['momentum_conv2'] = BETA1 * optimizer['momentum_conv2'] + ((1 - BETA1) * conv2_delta_w)

        optimizer['velocity_w1'] = BETA2 * optimizer['velocity_w1'] + ((1 - BETA2) * (d_fc1_w ** 2))
        optimizer['velocity_w2'] = BETA2 * optimizer['velocity_w2'] + ((1 - BETA2) * (d_fc2_w ** 2))
        optimizer['velocity_b0'] = BETA2 * optimizer['velocity_b0'] + ((1 - BETA2) * (d_fc1_b ** 2))
        optimizer['velocity_b1'] = BETA2 * optimizer['velocity_b1'] + ((1 - BETA2) * (d_fc2_b ** 2))
        optimizer['velocity_conv1'] = BETA2 * optimizer['velocity_conv1'] + ((1 - BETA2) * (conv1_delta_w ** 2))
        optimizer['velocity_conv2'] = BETA2 * optimizer['velocity_conv2'] + ((1 - BETA2) * (conv2_delta_w ** 2))

        # Corrections
        momentum_w1_corr = optimizer['momentum_w1'] / (1 - (BETA1 ** n_weight_updates))
        momentum_w2_corr = optimizer['momentum_w2'] / (1 - (BETA1 ** n_weight_updates))
        momentum_b0_corr = optimizer['momentum_b0'] / (1 - (BETA1 ** n_weight_updates))
        momentum_b1_corr = optimizer['momentum_b1'] / (1 - (BETA1 ** n_weight_updates))
        momentum_conv1_corr = optimizer['momentum_conv1'] / (1 - (BETA1 ** n_weight_updates))
        momentum_conv2_corr = optimizer['momentum_conv2'] / (1 - (BETA1 ** n_weight_updates))

        velocity_w1_corr = optimizer['velocity_w1'] / (1 - (BETA2 ** n_weight_updates))
        velocity_w2_corr = optimizer['velocity_w2'] / (1 - (BETA2 ** n_weight_updates))
        velocity_b0_corr = optimizer['velocity_b0'] / (1 - (BETA2 ** n_weight_updates))
        velocity_b1_corr = optimizer['velocity_b1'] / (1 - (BETA2 ** n_weight_updates))
        velocity_conv1_corr = optimizer['velocity_conv1'] / (1 - (BETA2 ** n_weight_updates))
        velocity_conv2_corr = optimizer['velocity_conv2'] / (1 - (BETA2 ** n_weight_updates))

        weights['conv1_w'] = weights['conv1_w'] - (
                LEARNING_RATE * (momentum_conv1_corr / (np.sqrt(velocity_conv1_corr) + EPS)))
        weights['conv2_w'] = weights['conv2_w'] - (
                LEARNING_RATE * (momentum_conv2_corr / (np.sqrt(velocity_conv2_corr) + EPS)))
        weights['fc1_w'] = weights['fc1_w'] - (LEARNING_RATE * (momentum_w1_corr / (np.sqrt(velocity_w1_corr) + EPS)))
        weights['fc2_w'] = weights['fc2_w'] - (LEARNING_RATE * (momentum_w2_corr / (np.sqrt(velocity_w2_corr) + EPS)))
        weights['fc1_b'] = weights['fc1_b'] - (LEARNING_RATE * (momentum_b0_corr / (np.sqrt(velocity_b0_corr) + EPS)))
        weights['fc2_b'] = weights['fc2_b'] - (LEARNING_RATE * (momentum_b1_corr / (np.sqrt(velocity_b1_corr) + EPS)))

    elif OPTIMIZER == 'MOMENTUM':
        optimizer['velocity_w1'] = BETA1 * optimizer['velocity_w1'] - (LEARNING_RATE * d_fc1_w)
        optimizer['velocity_w2'] = BETA1 * optimizer['velocity_w2'] - (LEARNING_RATE * d_fc2_w)
        optimizer['velocity_b0'] = BETA1 * optimizer['velocity_b0'] - (LEARNING_RATE * d_fc1_b)
        optimizer['velocity_b1'] = BETA1 * optimizer['velocity_b1'] - (LEARNING_RATE * d_fc2_b)
        optimizer['velocity_conv1'] = BETA1 * optimizer['velocity_conv1'] - (LEARNING_RATE * conv1_delta_w)
        optimizer['velocity_conv2'] = BETA1 * optimizer['velocity_conv2'] - (LEARNING_RATE * conv2_delta_w)

        weights['conv1_w'] = weights['conv1_w'] + optimizer['velocity_conv1']
        weights['conv2_w'] = weights['conv2_w'] + optimizer['velocity_conv2']
        weights['fc1_w'] = weights['fc1_w'] + optimizer['velocity_w1']
        weights['fc2_w'] = weights['fc2_w'] + optimizer['velocity_w2']
        weights['fc1_b'] = weights['fc1_b'] + optimizer['velocity_b0']
        weights['fc2_b'] = weights['fc2_b'] + optimizer['velocity_b1']

    else:
        weights['fc2_w'] = weights['fc2_w'] - LEARNING_RATE * d_fc2_w
        weights['fc2_b'] = weights['fc2_b'] - LEARNING_RATE * d_fc2_b

        weights['fc1_w'] = weights['fc1_w'] - LEARNING_RATE * d_fc1_w
        weights['fc1_b'] = weights['fc1_b'] - LEARNING_RATE * d_fc1_b

        weights['conv1_w'] = weights['conv1_w'] - LEARNING_RATE * conv1_delta_w
        weights['conv2_w'] = weights['conv2_w'] - LEARNING_RATE * conv2_delta_w

    return weights, optimizer


def train_model(train_images, train_labels,
                valid_images, valid_labels, epochs):

    print('##############################')
    print('# TRAIN MODEL')
    print('##############################')

    # Flag indicating if the validation is required
    validation_required = valid_images is not None

    # ##############################
    # CREATE BATCHES
    # ##############################

    # TRAIN
    train_images_batches = np.split(train_images, np.arange(BATCH_SIZE, len(train_images), BATCH_SIZE))
    train_images_labels = np.split(train_labels, np.arange(BATCH_SIZE, len(train_labels), BATCH_SIZE))

    # VALIDATION
    if validation_required:
        val_images_batches = np.split(valid_images, np.arange(BATCH_SIZE, len(valid_images), BATCH_SIZE))
        val_images_labels = np.split(valid_labels, np.arange(BATCH_SIZE, len(valid_labels), BATCH_SIZE))

    # Init model weights and optimizer parameters
    weights = init_model_weights()
    optimizer = init_optimizer_dictionary()

    # Number of times the weights are updated (needed for ADAM)
    n_weight_updates = 1

    # Keep track of the current best valid accuracy
    best_valid_acc = 0
    for e in range(epochs):
        print(f'=== EPOCH {e} ===')
        print('Epoch {} started...'.format(e))
        start = timer()

        # ##############################
        # TRAIN STEP
        # ##############################
        train_batch_loss = 0
        train_batch_acc = 0
        train_samples = 0

        for idx, input_data in enumerate(train_images_batches):
            train_samples += input_data.shape[0]

            input_labels = train_images_labels[idx]
            # One hot encoding
            input_labels_one_hot_encoded = np.zeros((input_labels.size, 10))
            for i in range(input_labels.shape[0]):
                position = input_labels[i]
                input_labels_one_hot_encoded[i, position] = 1

            scores, cache, loss, acc = forward(input_data, input_labels, input_labels_one_hot_encoded, weights)
            train_batch_loss += loss
            train_batch_acc += acc

            weights, optimizer = backward(input_data, input_labels_one_hot_encoded,
                                          scores, cache, weights, optimizer, n_weight_updates)
            n_weight_updates += 1

        # ##############################
        # VALIDATION STEP
        # ##############################
        if validation_required:
            valid_batch_loss = 0
            valid_batch_acc = 0
            valid_samples = 0

            for idx, input_data in enumerate(val_images_batches):
                valid_samples += input_data.shape[0]

                input_labels = val_images_labels[idx]

                # One hot encoding
                input_labels_one_hot_encoded = np.zeros((input_labels.size, 10))
                for i in range(input_labels.shape[0]):
                    position = input_labels[i]
                    input_labels_one_hot_encoded[i, position] = 1

                scores, cache, loss, acc = forward(input_data, input_labels, input_labels_one_hot_encoded, weights)
                valid_batch_loss += loss
                valid_batch_acc += acc

        end = timer()

        print('\tTRAIN Accuracy: {:.3f}\tTRAIN Loss: {:.3f}'.
              format(train_batch_acc / train_samples, train_batch_loss / train_samples))

        if validation_required:
            print('\tVALID Accuracy: {:.3f}\tVALID Loss: {:.3f}'.
                  format(valid_batch_acc / valid_samples, valid_batch_loss / valid_samples))

        print(f"Epoch {e} completed in (s): {end - start}")

        # Save model weights if validation score is higher
        if validation_required:
            if (valid_batch_acc / valid_samples) > best_valid_acc:
                best_valid_acc = valid_batch_acc / valid_samples
                print(f'\n(Higher accuracy found. Saving weights...)')
                save_weights(weights, e + 1, VALIDATION_WEIGHTS_PATH)

        print()

    # Save the last training weights
    if not validation_required:
        print(f'Training on full dataset completed. Saving weights...')
        save_weights(weights, epochs, TRAIN_WEIGHTS_PATH)

    print()


def test_model(weights_path, test_images, test_labels):
    print('##############################')
    print('# TEST MODEL')
    print('##############################')

    # Split in batches
    test_images_batches = np.split(test_images, np.arange(BATCH_SIZE, len(test_images), BATCH_SIZE))
    test_images_labels = np.split(test_labels, np.arange(BATCH_SIZE, len(test_labels), BATCH_SIZE))

    # Load weights from file system
    weights, _ = load_weights(weights_path)

    # Metrics
    test_batch_loss = 0
    test_batch_acc = 0
    test_samples = 0

    for idx, input_data in enumerate(test_images_batches):
        test_samples += input_data.shape[0]

        input_labels = test_images_labels[idx]
        # One hot encoding
        input_labels_one_hot_encoded = np.zeros((input_labels.size, 10))
        for i in range(input_labels.shape[0]):
            position = input_labels[i]
            input_labels_one_hot_encoded[i, position] = 1

        scores, cache, loss, acc = forward(input_data, input_labels, input_labels_one_hot_encoded, weights)
        test_batch_loss += loss
        test_batch_acc += acc

    print('TEST Accuracy: {:.3f}\tTEST Loss: {:.3f}'.
          format(test_batch_acc / test_samples, test_batch_loss / test_samples))


def main():
    if USE_CIFAR_10:
        dataset = Cifar10()
    else:
        dataset = Mnist()

    train_images, train_labels, \
        validation_images, validation_labels, \
        test_images, test_labels = \
        dataset.get_small_datasets() if TRAIN_SMALL_DATASET else dataset.get_datasets()

    # ######################################################################
    # TRAIN + VALIDATION
    # ######################################################################
    train_model(train_images, train_labels,
                validation_images, validation_labels, EPOCHS)

    # ######################################################################
    # TRAIN
    # ######################################################################
    # Load the number of epochs obtained after running the model on train
    # and validation set
    _, epochs = load_weights(VALIDATION_WEIGHTS_PATH)

    print(f'Best epochs loaded from file system: {epochs[0]-1}')
    print()

    train_model(np.concatenate((train_images, validation_images)),
                np.concatenate((train_labels, validation_labels)),
                None, None, epochs[0])

    # ######################################################################
    # TEST
    # ######################################################################
    test_model(TRAIN_WEIGHTS_PATH, test_images, test_labels)

    print()


if __name__ == '__main__':
    main()
