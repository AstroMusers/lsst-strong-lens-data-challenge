"""Shared Keras model for LSST strong-lens classification."""

import keras
from tensorflow.keras import layers


def make_model(input_shape, num_classes):
    """Build the classifier, returning one logit for binary classification."""
    inputs = keras.Input(shape=input_shape)
    num_bands = input_shape[2]

    x = layers.Conv2D(128, num_bands, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    previous_block_activation = x

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, num_bands, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, num_bands, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(num_bands, strides=2, padding="same")(x)

        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])
        previous_block_activation = x

    x = layers.SeparableConv2D(1024, num_bands, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        units = 1
    else:
        units = num_classes
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(units, activation=None)(x)
    return keras.Model(inputs, outputs)