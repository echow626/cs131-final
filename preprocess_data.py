import itertools
import numpy as np
import json
from collections import defaultdict
from hog import *
import tensorflow as tf


def preprocess_data(data_file, name):
    X = []
    y = []

    # Load training dataset
    with open(data_file, "r") as read_file:
        train_images = json.load(read_file)
    for image_obj in train_images:
        X.append(hog_feature("images/" + image_obj["image"]))
        y.append(np.ravel(image_obj["joints"]))
    X = np.column_stack((itertools.zip_longest(*X, fillvalue=0)))
    y = np.asarray(y)

    return X, y


# X_train, y_train = preprocess_data("filtered_data/single_person_train.json", "train")
# X_test, y_test = preprocess_data("filtered_data/single_person_test.json", "test")
# X_val, y_val = preprocess_data("filtered_data/single_person_valid.json", "valid")
X_val, y_val = preprocess_data("filtered_data/validation.json", "valid")
print(X_val, y_val)

np.save(f"tf_data/validation_X.npy", X_val)
np.save(f"tf_data/validation_y.npy", y_val)