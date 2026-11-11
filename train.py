# %%
import os
import time
import math
import yaml
from glob import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import keras
# from keras import layers
import tensorflow as tf
from tensorflow import data as tf_data
from tensorflow.keras import layers
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from pprint import pprint
from package_npzs import read_dataset_archive
from model import make_model
from repository_paths import get_model_output_path, get_repository_path, resolve_repository_path

print("Number of available GPUs: ", len(tf.config.list_physical_devices('GPU')))

# Read configuration and dataset paths relative to the configured repository root.
repository_path = get_repository_path()
config_path = repository_path / 'config.yml'
print(f'Reading from {config_path}...')
with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    print("No GPUs found.")
else:
    try:
        tf.config.set_visible_devices([gpus[0]], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("Using GPU 0.")
    except RuntimeError as error:
        print("Failed to set visible GPU:", error)

print("Visible GPUs:", tf.config.list_physical_devices('GPU'))

data_dir = resolve_repository_path(config['data_dir'])
dataset_path = data_dir / 'dataset.npz'
print(f'Reading from {dataset_path}...')
image_datasets = read_dataset_archive(str(dataset_path))
hsc_lenses = image_datasets['hsc_lens']
hsc_nonlenses = image_datasets['hsc_nonlens']
slsim_lenses = image_datasets['slsim_lens']
slsim_nonlenses = image_datasets['slsim_nonlens']

print(f'hsc_lens: {hsc_lenses.shape}')
print(f'hsc_nonlens: {hsc_nonlenses.shape}')
print(f'slsim_lens: {slsim_lenses.shape}')
print(f'slsim_nonlens: {slsim_nonlenses.shape}')

fig, axes = plt.subplots(2, 2, figsize=(11.4, 12), constrained_layout=True)
datasets = [
    (slsim_lenses, "slsim_lenses"),
    (hsc_lenses, "hsc_lenses"),
    (slsim_nonlenses, "slsim_nonlenses"),
    (hsc_nonlenses, "hsc_nonlenses"),
]

for ax, (images, title) in zip(axes.flat, datasets):
    grid_size = min(25, len(images))
    grid_rows = grid_cols = int(np.ceil(np.sqrt(grid_size)))
    for i in range(grid_size):
        row = i // grid_cols
        col = i % grid_cols
        sub_ax = ax.inset_axes([col/grid_cols, 1-row/grid_rows-1/grid_rows, 1/grid_cols, 1/grid_rows])
        sub_ax.imshow(images[i][:,:,:3])
        sub_ax.axis("off")  # Hide axes for each image
    ax.set_title(title)
    ax.axis("off")  # Hide main axes

plt.suptitle('Dataset Sample')
plt.show()

# %% [markdown]
# Combine the four image classes and assign binary lens labels.

# %%
data = np.concatenate([hsc_lenses, slsim_lenses, hsc_nonlenses, slsim_nonlenses], axis=0)
labels = np.array(([1] * (len(hsc_lenses) + len(slsim_lenses))) + ([0] * (len(hsc_nonlenses) + len(slsim_nonlenses))), dtype=np.uint8)
print(data.shape)
print(labels.shape)

# %% [markdown]
# Create a TensorFlow dataset from the image arrays and labels.

# %%
ds = tf_data.Dataset.from_tensor_slices((data, labels))
ds = ds.shuffle(buffer_size=len(labels), reshuffle_each_iteration=False)

# %% [markdown]
# Shuffle the rows once, then split them into about 95% training and 5% validation. The split stays fixed across epochs, while the training order is reshuffled each epoch.

# %%
# Calculate split sizes
total_size = len(labels)
train_size = int(0.95 * total_size)
val_size = total_size - train_size
print(f'end={total_size}, train={train_size}, val={val_size}')

# Split the dataset
train_ds = ds.take(train_size)
val_ds = ds.skip(train_size).take(val_size)
train_ds = train_ds.shuffle(buffer_size=train_size, reshuffle_each_iteration=True)

print(f"Train size: {train_ds.cardinality()}, Val size: {val_ds.cardinality()}")

# %% [markdown]
# Training images receive random horizontal and vertical flips and rotations. These transformations expose the classifier to varied image orientations.

# %%
data_augmentation_layers = [
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.25),
]

def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images

# Apply augmentation to training rows only.
train_ds = train_ds.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=tf_data.AUTOTUNE,
)

# %%
# Prefetch upcoming batches to overlap input preparation with model execution.
train_ds = train_ds.batch(256).prefetch(tf_data.AUTOTUNE)
val_ds = val_ds.batch(256).prefetch(tf_data.AUTOTUNE)

# %%
# def make_model(input_shape, num_classes):
#     inputs = layers.Input(shape=input_shape)

#     # Entry block
#     x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(inputs)
#     x = layers.BatchNormalization()(x)
#     x = layers.MaxPooling2D((2, 2))(x)

#     x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.MaxPooling2D((2, 2))(x)

#     x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.MaxPooling2D((2, 2))(x)

#     x = layers.Conv2D(512, (3, 3), padding="same", activation="relu")(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.GlobalAveragePooling2D()(x)

#     x = layers.Dropout(0.5)(x)
#     if num_classes == 2:
#         units = 1
#     else:
#         units = num_classes

#     outputs = layers.Dense(units, activation=None)(x)
    
#     return models.Model(inputs, outputs)

# %%
# def make_model(input_shape, num_classes, l2=1e-4):
#     inputs = keras.Input(shape=input_shape)
#     reg = keras.regularizers.l2(l2)

#     def conv_bn_act(x, filters, kernel_size=3, strides=1):
#         x = layers.Conv2D(filters, kernel_size, strides=strides, padding="same",
#                           kernel_regularizer=reg, use_bias=False)(x)
#         x = layers.BatchNormalization()(x)
#         return layers.Activation("relu")(x)

#     def bottleneck_block(x, filters, strides=1):
#         # Bottleneck: 1x1 reduce -> 3x3 -> 1x1 expand
#         shortcut = x
#         reduced = max(1, filters // 4)

#         x = layers.Conv2D(reduced, 1, strides=strides, padding="same",
#                           kernel_regularizer=reg, use_bias=False)(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.Activation("relu")(x)

#         x = layers.Conv2D(reduced, 3, strides=1, padding="same",
#                           kernel_regularizer=reg, use_bias=False)(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.Activation("relu")(x)

#         x = layers.Conv2D(filters, 1, strides=1, padding="same",
#                           kernel_regularizer=reg, use_bias=False)(x)
#         x = layers.BatchNormalization()(x)

#         # projection if shape/stride differs
#         if strides != 1 or shortcut.shape[-1] != filters:
#             shortcut = layers.Conv2D(filters, 1, strides=strides, padding="same",
#                                      kernel_regularizer=reg, use_bias=False)(shortcut)
#             shortcut = layers.BatchNormalization()(shortcut)

#         x = layers.add([x, shortcut])
#         return layers.Activation("relu")(x)

#     # Stem (kept gentle because input is small)
#     x = layers.Conv2D(64, 3, strides=1, padding="same", kernel_regularizer=reg, use_bias=False)(inputs)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation("relu")(x)
#     x = layers.MaxPooling2D(2, strides=2, padding="same")(x)

#     # Residual stages (filters, blocks, first_block_stride)
#     stages = [
#         (128, 2, 1),
#         (256, 2, 2),
#         (512, 2, 2),
#     ]
#     for filters, blocks, first_stride in stages:
#         x = bottleneck_block(x, filters, strides=first_stride)
#         for _ in range(blocks - 1):
#             x = bottleneck_block(x, filters, strides=1)

#     # Head
#     x = layers.GlobalAveragePooling2D()(x)
#     x = layers.Dropout(0.4)(x)
#     units = 1 if num_classes == 2 else num_classes
#     outputs = layers.Dense(units, activation=None, kernel_regularizer=reg)(x)

#     return keras.Model(inputs, outputs, name="resnet_like_v1")

# %% [markdown]
# Build the shared classifier.

# %%
model = make_model(input_shape=(41, 41, 5), num_classes=2)
# keras.utils.plot_model(model, show_shapes=True)

# %% [markdown]
# Train the model

# %%
epochs = 100

# ---- LR schedule (epoch-based warmup + cosine) ----
lr_max = 1e-3
lr_min = 1e-6
warmup_epochs = 5

def warmup_cosine(epoch, lr):
    if epoch < warmup_epochs:
        # linear warmup from 0 -> lr_max
        return lr_max * (epoch + 1) / float(warmup_epochs)
    # cosine decay from lr_max -> lr_min over remaining epochs
    progress = (epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * min(1.0, progress)))

lr_scheduler = keras.callbacks.LearningRateScheduler(warmup_cosine, verbose=0)

# training
model = make_model(input_shape=(41, 41, 5), num_classes=2)
# keras.utils.plot_model(model, show_shapes=True)

start = time.time()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr_max),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy(name="acc", threshold=0.0)],
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=6,
    restore_best_weights=True,
    verbose=1,
)

callbacks = [
    # early_stopping, 
    lr_scheduler
    ]

history = model.fit(
    train_ds,
    epochs=epochs,
    validation_data=val_ds,
    callbacks=callbacks,
)

end = time.time()
print(f"Training time: {end - start:.2f} seconds")

# %% [markdown]
# Compare loss and accuracy on the fixed training and validation rows. A widening gap can signal overfitting. Persistently poor results on both sets can signal underfitting.

# %%
loss = history.history['loss']
val_loss = history.history['val_loss']
accuracy = history.history['acc']
val_accuracy = history.history['val_acc']

_, ax = plt.subplots(1, 2, figsize=(7, 3), constrained_layout=True)
epoch_list = range(1, len(loss) + 1)

ax[0].plot(epoch_list, loss, 'bo-', label='Training')
ax[0].plot(epoch_list, val_loss, 'ro-', label='Validation')
ax[0].set_title('Loss')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
# ax[0].set_ylim(0, 1)
ax[0].legend()

ax[1].plot(epoch_list, accuracy, 'bo-', label='Training')
ax[1].plot(epoch_list, val_accuracy, 'ro-', label='Validation')
ax[1].set_title('Accuracy')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy')
# ax[1].set_ylim(0, 1)
ax[1].legend()

plt.show()

# %%
model_path = get_model_output_path('v4_retrain.keras')
print(f'Writing to {model_path.parent}...')
model_path.parent.mkdir(parents=True, exist_ok=True)
print(f'Writing to {model_path}...')
model.save(str(model_path))
