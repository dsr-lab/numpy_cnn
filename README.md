# CNN from scratch using only Numpy
I developed this project because I wanted to deeply understand the mathematics behind the learning process of a neural network.

The architecture is based on a VGG block, and the goal is to perform image classification.

![architecture](./plots/architecture.png) 

The model has been trained both on CIFAR10 and MNIST.

In the *config.py* it is possible to set variables that allow to change some model parameters (e.g.: dataset, optimizer, convolution type, ...).

To run the project just start the *main.py* file.

## TODO
* Add bias to convolutional layers
* complete the documentation by adding examples for better understanding the math
* remove some configuration flags (used during for the project presentation)
* remove math sample from comments code (used during for the project presentation)
