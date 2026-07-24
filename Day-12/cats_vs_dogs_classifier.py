import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

print("===== CATS VS DOGS CLASSIFIER (TRANSFER LEARNING) =====\n")

IMG_SIZE = 160
BATCH_SIZE = 32
EPOCHS = 10

# 1. Load and preprocess the dataset
#
# Note: I originally loaded this dataset using tensorflow_datasets (tfds.load),
# as suggested in the task. However, I ran into a known Windows-specific bug
# in tensorflow_datasets: zip archives always use forward slashes internally,
# but on Windows the library builds its file lookup path using backslashes,
# so it fails to find images that are genuinely present in the archive. This
# is a library bug, not something in my code, and it does not happen on
# Mac/Linux. I switched to downloading the dataset directly and loading it
# from folders instead, which avoids the buggy zip-parsing path entirely -
# this is the same approach used in TensorFlow's own official tutorial for
# this dataset.

dataset_url = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
path_to_zip = tf.keras.utils.get_file("cats_and_dogs.zip", origin=dataset_url, extract=True)

# I don't assume a fixed path for where "Cat" and "Dog" folders end up after
# extraction, since that depends on the zip's internal structure. Instead, I
# search the extraction directory for the folder that actually contains them.
extraction_root = os.path.dirname(path_to_zip)
data_dir = None
for root, dirs, files in os.walk(extraction_root):
    if "Cat" in dirs and "Dog" in dirs:
        data_dir = root
        break
if data_dir is None:
    raise FileNotFoundError(
        f"Could not find Cat/Dog folders anywhere under {extraction_root}. "
        "Check that the zip extracted correctly."
    )
print(f"Found dataset at: {data_dir}")

# Remove any corrupted (non-JFIF) images, which would otherwise crash
# training partway through on a bad file.
num_skipped = 0
for folder_name in ("Cat", "Dog"):
    folder_path = os.path.join(data_dir, folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            with open(fpath, "rb") as f:
                is_jfif = b"JFIF" in f.peek(10)
        except Exception:
            is_jfif = False
        if not is_jfif:
            num_skipped += 1
            os.remove(fpath)
print(f"Removed {num_skipped} corrupted images.")

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
)

class_names = train_ds.class_names
print(f"Class names: {class_names}")

# Keep a small raw (unprocessed) sample aside for displaying real-looking
# sample images later, since preprocess_input shifts pixel values in a way
# that isn't directly viewable as a normal image.
sample_images_raw = []
sample_labels_raw = []
for images, labels in val_ds.take(1):
    for i in range(8):
        sample_images_raw.append(images[i].numpy().astype("uint8"))
        sample_labels_raw.append(labels[i].numpy())

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
train_ds_processed = train_ds.map(lambda x, y: (preprocess_input(x), y)).prefetch(tf.data.AUTOTUNE)
val_ds_processed = val_ds.map(lambda x, y: (preprocess_input(x), y)).prefetch(tf.data.AUTOTUNE)

# 2. Build the model: MobileNetV2 backbone + custom classification head
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
)

# Freeze the pre-trained base - I reuse its already-learned features
# (learned from millions of ImageNet images) instead of retraining them,
# which would need far more data and time than I have for this project.
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(name="global_avg_pool"),
    layers.Dense(128, activation="relu", name="dense_head"),
    layers.Dropout(0.2, name="dropout"),
    layers.Dense(1, activation="sigmoid", name="output_layer"),
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# 3. Train the model
print("\nTraining the model...")
history = model.fit(
    train_ds_processed,
    validation_data=val_ds_processed,
    epochs=EPOCHS,
    verbose=2,
)

# 4. Evaluate on the validation dataset
val_loss, val_accuracy = model.evaluate(val_ds_processed, verbose=0)
final_train_accuracy = history.history["accuracy"][-1]

print(f"\nFinal Training Accuracy: {final_train_accuracy:.4f}")
print(f"Final Validation Accuracy: {val_accuracy:.4f}")
print(f"Final Validation Loss: {val_loss:.4f}")

if val_accuracy >= 0.93:
    print("Target validation accuracy (93%+) achieved.")
elif val_accuracy >= 0.90:
    print("Minimum validation accuracy (90%+) achieved, below the 93% target.")
else:
    print("Below the 90% minimum target - see README for experiments to try next.")

# 5. Sample predictions
sample_batch_images = np.array(sample_images_raw).astype("float32")
sample_batch_images_processed = preprocess_input(sample_batch_images)
predictions = model.predict(sample_batch_images_processed, verbose=0)
predicted_labels = (predictions.flatten() > 0.5).astype(int)

print("\nSample Predictions vs Actual Labels:")
for i in range(8):
    print(f"Actual: {class_names[sample_labels_raw[i]]:5} | Predicted: {class_names[predicted_labels[i]]}")

# 6. Plot training and validation accuracy and loss curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history["accuracy"], label="Training Accuracy")
axes[0].plot(history.history["val_accuracy"], label="Validation Accuracy")
axes[0].set_title("Training vs Validation Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()

axes[1].plot(history.history["loss"], label="Training Loss")
axes[1].plot(history.history["val_loss"], label="Validation Loss")
axes[1].set_title("Training vs Validation Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()

plt.tight_layout()
plt.savefig("training_accuracy_loss.png")
plt.close()
print("\nAccuracy and loss curves saved as training_accuracy_loss.png")

# 7. Display sample prediction images with actual vs predicted labels
plt.figure(figsize=(14, 6))
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(sample_images_raw[i])
    actual = class_names[sample_labels_raw[i]]
    predicted = class_names[predicted_labels[i]]
    color = "green" if predicted_labels[i] == sample_labels_raw[i] else "red"
    plt.title(f"Actual: {actual}\nPredicted: {predicted}", color=color, fontsize=10)
    plt.axis("off")
plt.tight_layout()
plt.savefig("sample_predictions.png")
plt.close()
print("Sample prediction images saved as sample_predictions.png")