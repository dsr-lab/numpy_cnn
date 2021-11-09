EPOCHS = 15
BATCH_SIZE = 128
CONV_PADDING = 0

USE_CIFAR_10 = False
TRAIN_SMALL_DATASET = False  # Just try on dummy dataset
USE_HE_WEIGHT_INITIALIZATION = True  # True for best results

USE_FAST_CONV = True
USE_DROPOUT = False
CONV_DROPOUT_PROBABILITY = 0.9
DENSE_DROPOUT_PROBABILITY = 0.8


LEARNING_RATE = 1e-3
EPS = 1e-8
BETA1 = 0.9
BETA2 = 0.999
OPTIMIZER = 'ADAM'  # Valid values: ADAM, MOMENTUM

TRAIN_WEIGHTS_PATH = 'weights/cifar10_train' if USE_CIFAR_10 else 'weights/mnist_train'
VALIDATION_WEIGHTS_PATH = 'weights/cifar10_validation' if USE_CIFAR_10 else 'weights/mnist_validation'




