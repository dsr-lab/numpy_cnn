import os
import pickle
import tarfile
import urllib.request
import numpy as np


class Cifar10:

    def __init__(self):

        self.train_images = None
        self.test_images = None
        self.train_labels = None
        self.test_labels = None
        self.validation_images = None
        self.validation_labels = None

        self.classes = (
            'plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse',
            'ship', 'truck'
        )

        self.__init_dataset()

    def __init_dataset(self):

        dataset_path = self.create_dataset_path()

        self.download_dataset(dataset_path)

        self.process_dataset(dataset_path)

    @staticmethod
    def create_dataset_path():
        dataset_path = os.path.join(os.path.dirname(__file__), "cifar10")
        os.makedirs(dataset_path, exist_ok=True)
        return dataset_path

    def download_dataset(self, dataset_path):

        # Official url taken from https://www.cs.toronto.edu/~kriz/cifar.html
        url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

        # Tar file properties
        tar_file_name = url.split("/")[-1]
        tar_path = os.path.join(dataset_path, tar_file_name)

        if tar_file_name not in os.listdir(dataset_path):
            print('Downloading Cifar 10...')
            _, _ = urllib.request.urlretrieve(url,
                                              filename=tar_path,
                                              reporthook=self.__download_progress)
            print('Download complete')

            # Extract the downloaded file
            with tarfile.open(tar_path) as tar_object:
                tar_object.extractall(dataset_path)

    def process_dataset(self, dataset_path):
        # The tar contains some files that must be ignored
        # (e.g.: readme.html).
        # Create a list of files that must be processed for
        # creating the final dataset.
        expected_files = ['cifar-10-batches-py/data_batch_1',
                          'cifar-10-batches-py/data_batch_2',
                          'cifar-10-batches-py/data_batch_3',
                          'cifar-10-batches-py/data_batch_4',
                          'cifar-10-batches-py/data_batch_5',
                          'cifar-10-batches-py/test_batch']
        # There are 60000 images. Each image is 3x32x32 = 3072
        # There are 60000 labels
        images = np.zeros(shape=(60000, 3, 32, 32))
        labels = np.zeros(shape=(60000,))
        # Process each file and append images and labels to the
        # correct array
        for idx, file_name in enumerate(expected_files):
            file = os.path.join(dataset_path, file_name)
            with open(file, 'rb') as fo:
                dictionary = pickle.load(fo, encoding='bytes')

                file_images = dictionary.get(b'data')
                file_labels = dictionary.get(b'labels')

                file_images = file_images / 255
                file_images = file_images.reshape(10000, 3, 32, 32)

                images[idx * 10000:10000 * (idx + 1)] = file_images
                labels[idx * 10000:10000 * (idx + 1)] = file_labels
        # Split into train and test
        self.train_images, self.test_images = images[:50000], images[50000:]
        self.train_labels, self.test_labels = labels[:50000], labels[50000:]

        # Shuffle the train set
        permutation_indices = np.random.permutation(len(self.train_images))
        self.train_images = self.train_images[permutation_indices]
        self.train_labels = self.train_labels[permutation_indices]

        # Normalize with mean and std
        mean_train = self.train_images.mean(axis=(0, 2, 3), keepdims=True)
        std_train = self.train_images.std(axis=(0, 2, 3), keepdims=True)
        self.train_images = (self.train_images - mean_train) / std_train

        mean_test = self.test_images.mean(axis=(0, 2, 3), keepdims=True)
        std_test = self.test_images.std(axis=(0, 2, 3), keepdims=True)
        self.test_images = (self.test_images - mean_test) / std_test

        # Create the validation set
        self.validation_images = self.train_images[45000:]
        self.validation_labels = self.train_labels[45000:]

        self.train_images = self.train_images[:45000]
        self.train_labels = self.train_labels[:45000]

        a = self.train_labels
        print()

    @staticmethod
    def __download_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percentage = downloaded * 100 / total_size
        print('\r Download progress: {:.2f}%'.format(percentage), end='', flush=True)

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
