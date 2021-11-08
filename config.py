BATCH_SIZE = 128
EPOCHS = 1
CONV_DROPOUT_PROBABILITY = 0.9
DENSE_DROPOUT_PROBABILITY = 0.8
CONV_PADDING = 0

OPTIMIZER = 'ADAM'  # Valid values: ADAM, MOMENTUM

TRAIN_SMALL_DATASET = True
USE_CIFAR_10 = True

USE_FAST_CONV = True
USE_DROPOUT = False


LEARNING_RATE = 1e-3

EPS = 1e-8
BETA1 = 0.9
BETA2 = 0.999

TRAIN_WEIGHTS_PATH = 'weights/train'
VALIDATION_WEIGHTS_PATH = 'weights/validation'
