""" Project Information """
__author__ = "Aaron Flanagan"
__copyright__ = "NUIG"
__credits__ = "Aaron Flanagan"
__version__ = "1.0.0"
__email__ = "A.flanagan18@nuigalway.com"
__status__ = "Development"

import numpy as np
from matplotlib import animation, pyplot as plt
import h5py
import imageio

"""
Script to import .mat binary files, process and export OCT scans as .pngs
"""
filename = "RightRingGray"

# LOAD OCT DATA
X = np.asarray(h5py.File('G:/MSC Capstone/image binaries/telesto/' + filename + '.mat', 'r').get("FImage"))

# Fix the fromfile C-order unravel bug
X = np.moveaxis(X, 1, 0)
X = np.moveaxis(X, -1, 1)

# remove last blank images
X = X[:-1, ...]

# cut noise from end of images
X = X[:, :580, :]

# flip images 180 horizontal to match nsOCT
X = np.flip(X, axis=2)


# Displays array of images in a loop
def loop_images(img_arr):
    # for storing and displaying the animated images
    frames = []
    fig = plt.figure()

    plt.tick_params(labelsize=10)
    plt.xlabel('Lateral extent (mm)', fontsize=12)
    plt.ylabel('Depth (mm)', fontsize=12)

    # add images as frames to the plot
    for img in range(len(img_arr)):
        frames.append([plt.imshow(img_arr[img, ...], animated=True)])

    cbar = plt.colorbar()
    cbar.set_label('Grayscale intensity', fontsize=11)

    animation.ArtistAnimation(fig, frames, interval=40, blit=True, repeat=False)
    plt.show()


answer = input("Plot data? [Y/N]? ").lower()
if answer == "y":
    loop_images(X)

# stores data as 2 dim grayscale, min=0, max=255
for i in range(0, len(X)):
    imageio.imwrite("data/train/labels/" + str(i) + ".png", X[i, ...])
