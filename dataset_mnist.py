from keras.datasets import mnist
from utils import *


class Mnist:
    """
    Class that provides access to MNIST dataset
    """

    def __init__(self):

        self.train_images = None
        self.test_images = None
        self.train_labels = None
        self.test_labels = None
        self.validation_images = None
        self.validation_labels = None

        self.classes = (
            '0', '1', '2', '3',
            '4', '5', '6', '7',
            '8', '9'
        )

        self.__init_dataset()

    def __init_dataset(self):
        self.process_dataset()

    def process_dataset(self):
        (trainX, trainy), (testX, testy) = mnist.load_data()
        self.train_images = np.expand_dims(trainX, axis=1)
        self.train_images = self.train_images / 255
        self.train_images = (self.train_images - 0.5) / 0.5
        self.train_labels = trainy

        self.test_images = np.expand_dims(testX, axis=1)
        self.test_images = self.test_images / 255
        self.test_images = (self.test_images - 0.5) / 0.5
        self.test_labels = testy

        # Shuffle the train set
        #np.random.seed(31)
        permutation_indices = np.random.permutation(len(self.train_images))
        self.train_images = self.train_images[permutation_indices]
        self.train_labels = self.train_labels[permutation_indices]

        self.validation_images = self.train_images[55000:]
        self.validation_labels = self.train_labels[55000:]

        self.train_images = self.train_images[:55000]
        self.train_labels = self.train_labels[:55000]

    def get_small_datasets(self):
        return \
            self.train_images[:500], self.train_labels[:500],  \
            self.validation_images[:100], self.validation_labels[:100], \
            self.test_images[:100], self.test_labels[:100]

    def get_datasets(self):
        return \
            self.train_images, self.train_labels,  \
            self.validation_images, self.validation_labels, \
            self.test_images, self.test_labels
