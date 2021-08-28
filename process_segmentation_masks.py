""" Project Information """
__author__ = "Aaron Flanagan"
__copyright__ = "NUIG"
__credits__ = "Aaron Flanagan"
__version__ = "1.0.0"
__email__ = "A.flanagan18@nuigalway.com"
__status__ = "Development"


import numpy as np
from matplotlib import animation, pyplot as plt
import imageio

PATH = "data/train/labels/"

# load images, filter class labels and save to disk
for i in range(242, 301):
    img = plt.imread(PATH + str(i) + ".png")
    img = img[..., 0]
    img = np.where(img < 1, 0, img)
    imageio.imwrite(PATH + str(i) + ".png", img)


def loop_images(img_arr):
    # for storing and displaying the animated images
    frames = []
    fig = plt.figure()

    plt.tick_params(labelsize=10)
    plt.xlabel('Lateral extent (mm)', fontsize=12)
    plt.ylabel('Depth (mm)', fontsize=12)

    for i in range(len(img_arr)):
        frames.append([plt.imshow(img_arr[i, :, :], cmap="gray", animated=True)])

    cbar = plt.colorbar()
    cbar.set_label('Grayscale intensity', fontsize=11)

    animation.ArtistAnimation(fig, frames, interval=3, blit=True, repeat_delay=1000)
    plt.show()


answer = input("Plot data?? [Y/N]? ").lower()
if answer == "y":
    loop_images(X)


