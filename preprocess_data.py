import itertools
import numpy as np
import json
from collections import defaultdict
from hog import *
import tensorflow as tf

# 106191 --> size of hog on everything

def filter_for_single_person_data(data_filepath, save_to_filepath):
    with open(data_filepath, 'r') as file:
        loaded_data = json.load(file)
    
    image_to_data = defaultdict(list)
    for data in loaded_data:
        image_to_data[data["image"]].append(data)

    single_person_images = {key: data[0] for key, data in image_to_data.items() if len(data) == 1}

    single_person_json = [data for data in loaded_data if data["image"] in single_person_images.keys()]
    

    with open(save_to_filepath, 'w') as write_to:
        json.dump(single_person_json, write_to)
    print(f"Filtered data saved to {save_to_filepath}")

    return single_person_images

def preprocess_data(data_file, num_images=None):
    X = []
    y = []
    images = []

    # Load training dataset
    with open(data_file, "r") as read_file:
        train_images = json.load(read_file)
    if num_images is not None:
        train_images = train_images[:num_images]
    for image_obj in train_images:
        X.append(hog_feature("images/" + image_obj["image"]))
        y.append(np.ravel(image_obj["joints"]))
        images.append("images/" + image_obj["image"])
    X = np.column_stack((itertools.zip_longest(*X, fillvalue=0)))
    y = np.asarray(y)

    return X, y, images

X, y, images = preprocess_data("filtered_data/single_person_train.json")

# Split data
print(X, y)
print(X.shape)
print(y.shape)
data_len = X.shape[0]
num_train = np.floor(data_len * 0.8).astype(int)
num_test_val = np.floor(data_len * 0.1).astype(int)
# Split into test, train, valid
np.save(f"tf_data/train_X.npy", X[:num_train, :])
np.save(f"tf_data/train_y.npy", y[:num_train, :])
np.save(f"tf_data/test_X.npy", X[num_train:num_train+num_test_val, :])
np.save(f"tf_data/test_y.npy", y[num_train:num_train+num_test_val, :])
np.save(f"tf_data/valid_X.npy", X[num_train+num_test_val:, :])
np.save(f"tf_data/valid_y.npy", y[num_train+num_test_val:, :])
print("DONE!")