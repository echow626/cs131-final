import tensorflow as tf
import numpy as np
import json
from hog import *

X_train = []
y_train = []
X_val = []
y_val = []
X_test = []
y_test = []

def preprocess_data():
    global X_train, y_train, X_val, y_val, X_test, y_test

    with open("filtered_data/single_person_small_train.json", "r") as read_file:
        train_images = json.load(read_file)
    for image_obj in train_images:
        X_train.append(hog_feature("images/" + image_obj["image"]))
        y_train.append(np.ravel(image_obj["joints"]))
    X_train = tf.convert_to_tensor(tf.keras.utils.pad_sequences(X_train))

    print("Finished loading in training data")

    # # Load training dataset
    # with open("filtered_data/single_person_train.json", "r") as read_file:
    #     train_images = json.load(read_file)
    # for image_obj in train_images:
    #     X_train.append(hog_feature("images/" + image_obj["image"]))
    #     y_train.append(np.ravel(image_obj["joints"]))
    # X_train = tf.convert_to_tensor(tf.keras.utils.pad_sequences(X_train))

    # print("Finished loading in training data")
    
    # # Load test dataset
    # with open("filtered_data/single_person_test.json", "r") as read_file:
    #     test_images = json.load(read_file)
    # for image_obj in test_images:
    #     X_test.append(hog_feature("images/" + image_obj["image"]))
    #     y_test.append(np.ravel(image_obj["joints"]))
    # X_test = tf.convert_to_tensor(tf.keras.utils.pad_sequences(X_test))

    # print("Finished loading in test data")

    # # Load validation dataset
    # with open("filtered_data/single_person_valid.json", "r") as read_file:
    #     val_images = json.load(read_file)
    # for image_obj in val_images:
    #     X_val.append(hog_feature("images/" + image_obj["image"]))
    #     y_val.append(np.ravel(image_obj["joints"]))
    # X_val = tf.convert_to_tensor(tf.keras.utils.pad_sequences(X_val))

    # print("Finished loading in validation data")

def model():
    feature_length = None  # determine this from hog
    # Define the CNN architecture
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(feature_length,)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(32)  # Output layer for 16 coordinates (2D)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

    # Evaluate the model
    loss = model.evaluate(X_test, y_test)

    # Make predictions
    # predictions = model.predict(X_new_images)

# load_data()x