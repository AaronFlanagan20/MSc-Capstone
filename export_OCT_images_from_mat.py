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

filename = "RightRingGray"

# LOAD OCT DATA
X = h5py.File('G:\\MSC Capstone\\image binaries\\telesto\\' + filename + '.mat', 'r').get("FImage")
X = np.asarray(X)
# X = np.reshape(X, (1000, 1000, 1024))

# Fix the fromfile C-order unravel bug
X = np.moveaxis(X, 1, 0)
X = np.moveaxis(X, -1, 1)

# confirm last images is the blank
#plt.imshow(X[-1, ...], cmap="gray", extent=[0, 5, 3.6, 0])
#plt.show()

# remove last blank images
X = X[:-1, ...]

# cut noise from end of images
X = X[:, :580, :]

X = np.flip(X, axis=2)

def plot(X):
    # for storing and displaying the animated images
    frames = []
    fig = plt.figure()

    plt.tick_params(labelsize=10)
    plt.xlabel('Lateral extent (mm)', fontsize=12)
    plt.ylabel('Depth (mm)', fontsize=12)

    for i in range(len(X)):
        frames.append([plt.imshow(X[i, :, :], cmap="gray", animated=True)])

    cbar = plt.colorbar()
    cbar.set_label('Grayscale intensity', fontsize=11)

    animation.ArtistAnimation(fig, frames, interval=3, blit=True, repeat_delay=1000)
    plt.show()


answer = input("Plot data?? [Y/N]? ").lower()
if answer == "y":
    plot(X)

# stores data as 2 dim grayscale, min=0, max=255
for i in range(0, len(X)):
    imageio.imwrite("data/train/labels/" + str(i) + ".png", X[i, ...])
