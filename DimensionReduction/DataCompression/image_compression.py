#Grant White
#November 2019
"""
Data (image) Compression using Singular Value Decomposition (SVD).
"""

import numpy as np
from scipy import linalg as la
import math
from matplotlib import pyplot as plt
from imageio import imread
import SVD


def compress_image(filename, s):
    """Plots the original image found at 'filename' and the rank s approximation
    of the image found at 'filename.' States the number of entries used to store
    the original image and the approximation.

    Parameters:
        filename (str): Image file path.
        s (int): Rank of new image.
    """
    original = imread(filename) / 255

    #grayscale
    if len(original.shape) == 2:
        reduced, num = SVD.svd_approx(original, s)

        #plot
        plt.subplot(121)
        plt.imshow(original, cmap='gray')
        plt.axis('off')
        plt.title('Entries Stored: ' + str(original.size))

        plt.subplot(122)
        plt.imshow(reduced, cmap='gray')
        plt.axis('off')
        plt.title('Entries Stored: ' + str(num))

        plt.suptitle('Difference: ' + str(original.size - num))
        plt.show()

    #color
    else:
        R, num1 = SVD.svd_approx(original[:,:,0], s)
        G, num2 = SVD.svd_approx(original[:,:,1], s)
        B, num3 = SVD.svd_approx(original[:,:,2], s)

        R = np.clip(R, 0, 1)
        G = np.clip(G, 0, 1)
        B = np.clip(B, 0, 1)

        num = num1 + num2 + num3
        reduced = np.dstack((R, G, B))

        #plot
        plt.subplot(121)
        plt.imshow(original)
        plt.axis('off')
        plt.title('Entries Stored: ' + str(original.size))

        plt.subplot(122)
        plt.imshow(reduced)
        plt.axis('off')
        plt.title('Entries Stored: ' + str(num))

        plt.suptitle('Difference: ' + str(original.size - num))
        plt.show()


# compress_image('hubble.jpg', 200)
