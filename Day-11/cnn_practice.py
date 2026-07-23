import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import fashion_mnist

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

print("===== PRACTICE 1: LOAD, VISUALIZE, NORMALIZE =====\n")

# 1. Load the Fashion MNIST dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)

# 2. Visualize at least 10 sample images with their labels
plt.figure(figsize=(12, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train[i], cmap="gray")
    plt.title(class_names[y_train[i]])
    plt.axis("off")
plt.tight_layout()
plt.savefig("practice_sample_images.png")
plt.close()
print("10 sample images saved as practice_sample_images.png")

# 3. Normalize the dataset (0-255 -> 0-1)
X_train = X_train / 255.0
X_test = X_test / 255.0
print(f"Pixel range after normalization: {X_train.min()} to {X_train.max()}")

# CNNs expect a channel dimension (height, width, channels). Fashion MNIST
# images are grayscale, so we add a channel dimension of 1 -
# from (28, 28) to (28, 28, 1) - rather than 3 (which would be for RGB).
X_train_cnn = X_train.reshape(-1, 28, 28, 1)
X_test_cnn = X_test.reshape(-1, 28, 28, 1)
print("Reshaped for CNN input:", X_train_cnn.shape)

print("\n===== PRACTICE 2: BUILD AND TRAIN A SIMPLE CNN =====\n")

# 4. Build a simple CNN
model = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),

    # Convolution Layer: slides small filters (kernels) across the image to
    # detect local patterns like edges and textures. 32 filters, each 3x3.
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu", name="conv_layer"),

    # Max Pooling Layer: shrinks each feature map by keeping only the
    # strongest (maximum) value in each small region, reducing size while
    # keeping the most important detected features.
    layers.MaxPooling2D(pool_size=(2, 2), name="pool_layer"),

    # Flatten Layer: converts the 2D feature maps into a 1D vector so they
    # can be passed into standard Dense layers.
    layers.Flatten(name="flatten_layer"),

    # Dense (Fully Connected) Layer: learns to combine the extracted features
    # to make a classification decision.
    layers.Dense(64, activation="relu", name="dense_layer"),

    # Output Layer: 10 neurons (one per clothing category), Softmax gives a
    # probability distribution across all classes.
    layers.Dense(10, activation="softmax", name="output_layer"),
])

model.summary()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print("\nTraining the CNN (observe accuracy/loss improving each epoch)...")
history = model.fit(
    X_train_cnn, y_train,
    epochs=5,
    validation_split=0.2,
    verbose=2,
)

print("\n===== PRACTICE 3: EVALUATE THE MODEL =====\n")

# 5. Evaluate using Training Accuracy, Test Accuracy, Loss
train_accuracy = history.history["accuracy"][-1]
test_loss, test_accuracy = model.evaluate(X_test_cnn, y_test, verbose=0)

print(f"Final Training Accuracy: {train_accuracy:.4f}")
print(f"Final Test Accuracy: {test_accuracy:.4f}")
print(f"Final Test Loss: {test_loss:.4f}")

# 6. Predictions on sample images
predictions = model.predict(X_test_cnn[:5], verbose=0)
predicted_labels = np.argmax(predictions, axis=1)

print("\nSample Predictions vs Actual Labels:")
for i in range(5):
    print(f"Actual: {class_names[y_test[i]]:15} | Predicted: {class_names[predicted_labels[i]]}")