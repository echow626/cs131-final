from __future__ import print_function
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
from skimage import feature
from skimage import data, color, exposure
from skimage.transform import rescale, resize, downscale_local_mean
import glob, os
import fnmatch
import time
import warnings
warnings.filterwarnings('ignore')

def hog_feature(image, pixel_per_cell=8):
    """
    Compute hog feature for a given image.

    Important:
    - Use the hog function provided by skimage to generate both the
      feature vector and the visualization image.
    - For the block normalization parameter, use L1!

    Args:
        image: an image with object that we want to detect.
        pixel_per_cell: number of pixels in each cell, an argument for hog descriptor.

    Returns:
        hog_feature: a vector of hog representation.
        hog_image: an image representation of hog provided by skimage.
    """
    ### YOUR CODE HERE
    hog_feature, hog_image = feature.hog(image, pixels_per_cell=(pixel_per_cell, pixel_per_cell), block_norm='L1', visualize='True', feature_vector='True', transform_sqrt='True')
    ### END YOUR CODE
    return hog_feature, hog_image

def process_image(image_path='008963454_copy.jpg'):
	image = io.imread(image_path, as_gray=True)
	hog_ft, hog_img = hog_feature(image)
	print(hog_ft)
	print(hog_ft.shape)
	plt.imshow(hog_img)
	plt.show()

process_image('063271031_copy.jpg')