import os
import re
import matplotlib.pyplot as plt
import numpy as np


def get_train_val_values(file_name, file_ext='.txt'):
    epochs = []
    epochs_time = []
    train_accuracy = []
    validation_accuracy = []

    train_loss = []
    validation_loss = []

    idx = 0

    with open(file_name + file_ext) as file:
        for line in file:
            if 'TRAIN Accuracy:' in line:
                epochs.append(idx)
                idx += 1
                accuracy = re.search('TRAIN Accuracy:(.*)TRAIN Loss:', line).group(1).strip()
                loss = re.search('TRAIN Loss:(.*)', line).group(1).strip()

                train_accuracy.append(float(accuracy))
                train_loss.append(float(loss))

            elif 'VALID Accuracy:' in line:
                accuracy = re.search('VALID Accuracy:(.*)VALID Loss:', line).group(1).strip()
                loss = re.search('VALID Loss:(.*)', line).group(1).strip()

                validation_accuracy.append(float(accuracy))
                validation_loss.append(float(loss))
            elif 'completed in (s):' in line:
                time = re.search('completed in \(s\):(.*)', line).group(1).strip()

                epochs_time.append(float(time))

    return train_accuracy, train_loss, validation_accuracy, validation_loss, epochs, epochs_time


def get_file_name_from_path(path):
    return os.path.basename(path).split('_')
