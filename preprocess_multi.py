import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from skimage import io, feature, data, color, exposure

def preprocess(data_pt):
  img_file = get_img_file_path(data_pt[1])
  X = hog_feature(img_file)
  y = data_pt[2:-3]
  return X, y

def resize_name(img_file: str) -> str:
  return img_file[:-4] + "_resized.png"

def get_img_file_path(img_file: str) -> str:
  return f"resized_images/{resize_name(img_file)}"

def hog_feature(image_path):
  image = io.imread(image_path, as_gray=True)
  pixel_per_cell = min(image.shape[0]/25, image.shape[1]/25)
  hog_feature = feature.hog(image, pixels_per_cell=(pixel_per_cell, pixel_per_cell),
                            block_norm='L2-Hys', feature_vector=True, transform_sqrt=True)
  return hog_feature