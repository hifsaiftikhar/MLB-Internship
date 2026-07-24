import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

IMG_SIZE = 160
BATCH_SIZE = 32

print("===== PRACTICE 1: PRE-TRAINED MOBILENETV2 =====\n")

# 1. Load a pre-trained MobileNetV2 model.
# include_top=False drops MobileNetV2's original 1000-class ImageNet
# classifier, since I want to attach my own classification head instead.
# weights="imagenet" loads weights already learned from millions of images.
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
)

# 2. Explore its architecture
print(f"MobileNetV2 has {len(base_model.layers)} layers.")
print("\nFirst 5 layers:")
for layer in base_model.layers[:5]:
    print(f"  {layer.name} - {layer.__class__.__name__}")
print("\nLast 5 layers:")
for layer in base_model.layers[-5:]:
    print(f"  {layer.name} - {layer.__class__.__name__}")

print(f"\nTotal parameters: {base_model.count_params():,}")

# 3. Freeze the base model layers.
# I keep MobileNetV2's already-learned weights unchanged during training,
# so I reuse its learned feature extraction instead of retraining it from
# scratch (which would need far more data and time than I have here).
base_model.trainable = False
print(f"\nBase model trainable: {base_model.trainable}")

# 4. Add a custom classification head on top of the frozen base
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(name="global_avg_pool"),
    layers.Dense(128, activation="relu", name="dense_head"),
    layers.Dropout(0.2, name="dropout"),
    layers.Dense(1, activation="sigmoid", name="output_layer"),  # binary: cat vs dog
])

model.summary()

print("\n===== PRACTICE 2: LOAD AND PREPROCESS CATS VS DOGS =====\n")

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

# 1. Download the Cats vs Dogs dataset directly (Microsoft's original zip)
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

# The dataset contains a small number of corrupted image files that are not
# valid JPEGs. I filter these out first, otherwise loading crashes partway
# through training on a bad file.
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

# 2. Split into training (80%) and validation (20%) sets, resizing images
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

print("\nClass names:", train_ds.class_names)

# 3. Preprocess: scale pixel values the way MobileNetV2 expects
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y)).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y)).prefetch(tf.data.AUTOTUNE)

print("Images resized to", (IMG_SIZE, IMG_SIZE), "and preprocessed for MobileNetV2.")
print("Dataset ready, batched, and split into training and validation sets.")