""" Project Information """
__author__ = "Aaron Flanagan"
__copyright__ = "NUIG"
__credits__ = "Aaron Flanagan"
__version__ = "1.0.0"
__email__ = "A.flanagan18@nuigalway.com"
__status__ = "Development"

import keras
import matplotlib.pyplot as plt
import numpy as np
from unet import UNet

"""
Main script used to load the data and build the model
"""


def load_data(data_type, num_training_samples, num_test_samples, filename="RightRing"):
    path = "data/train/"
    all_samples = num_training_samples + num_test_samples

    # LOAD nsOCT DATA
    if data_type == "ns":
        X = np.load(path + 'images/' + filename + '.npy')

        # MIN-MAX NORMALISE DATA 0-1
        X = (X - np.min(X)) / (np.max(X) - np.min(X))

    # OCT DATA
    if data_type == "oct":
        # matplotlib normalises the data on import 0-255 > 0.0-1.0
        X = np.array([plt.imread("data/OCT/" + str(i) + ".png") for i in range(0, all_samples)])

    # add channel dimension
    X = X[..., np.newaxis]

    # LOAD TRAINING DATA
    X_train = X[:num_training_samples + 1, ...]
    X_test = X[num_training_samples + 1:all_samples, ...]

    # LOAD LABELS
    y = np.array([plt.imread(path + "labels/" + str(i) + ".png") for i in range(0, all_samples)])
    y_train = y[:num_training_samples + 1, ..., np.newaxis]
    y_test = y[num_training_samples + 1:all_samples, ..., np.newaxis]

    return X_train, y_train, X_test, y_test


X_train, y_train, X_test, y_test = load_data("oct", num_training_samples=300, num_test_samples=50)

# BUILD MODEL
model = UNet(X_train.shape[1:])
model.compile(optimizer=keras.optimizers.Adam(lr=0.01))

# RUN & SAVE MODEL
model_history = model.fit(X_train, y_train, epochs=10, batch_size=1)
model.save(model_history)

# EVALUATE TEST DATA
test_eval = model.evaluate(X_test, y_test, batch_size=1)

