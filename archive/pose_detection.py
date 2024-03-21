import tensorflow as tf
import numpy as np
import json
from hog import *

# 106191 --> hog features

def load_data():
    X_train = np.load("tf_data/train_X.npy")
    y_train = np.load("tf_data/train_y.npy")
    X_val = np.load("tf_data/valid_X.npy")
    y_val = np.load("tf_data/valid_y.npy")
    X_test = np.load("tf_data/test_X.npy")
    y_test = np.load("tf_data/test_y.npy")
    return X_train, y_train, X_val, y_val, X_test, y_test


def create_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='sigmoid', input_shape=(input_shape,), 
                              kernel_initializer=tf.keras.initializers.GlorotUniform(),
                              kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='sigmoid'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(32, activation='sigmoid'),
        tf.keras.layers.Dense(32)  # Output layer for 16 coordinates (2D)
    ])

    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test):
    input_shape = X_train.shape[1]
    model = create_model(input_shape)

    # Train the model
    model.fit(X_train, y_train, epochs=250, batch_size=128, validation_data=(X_val, y_val))

    # Evaluate the model on the test set
    loss = model.evaluate(X_test, y_test)
    print("Test Loss:", loss)

    return model

X_train, y_train, X_val, y_val, X_test, y_test = load_data()
trained_model = train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test)

