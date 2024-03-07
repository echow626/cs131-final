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

def hog_feature(image_path, pixel_per_cell=8):
    # hog_feature, hog_image = feature.hog(image, pixels_per_cell=(pixel_per_cell, pixel_per_cell), block_norm='L1', visualize='True', feature_vector='True', transform_sqrt='True')
    # return hog_feature, hog_image
    image = io.imread(image_path, as_gray=True)
    hog_feature = feature.hog(image, pixels_per_cell=(pixel_per_cell, pixel_per_cell), block_norm='L1', feature_vector='True', transform_sqrt='True')
    return hog_feature

# def process_image(image_path='008963454_copy.jpg'):
	# image = io.imread(image_path, as_gray=True)
    # return hog_feature(image)
	# hog_image_rescaled = exposure.rescale_intensity(hog_img, in_range=(0, 0.02))
	# print(hog_ft)
	# print(hog_ft.shape)
	# plt.imshow(hog_image_rescaled)
	# plt.show()
    

# process_image('063271031_copy.jpg')