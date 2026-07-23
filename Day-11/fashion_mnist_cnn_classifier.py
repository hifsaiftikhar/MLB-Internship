import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import fashion_mnist
from sklearn.metrics import confusion_matrix

print("===== FASHION MNIST IMAGE CLASSIFIER (CNN) =====\n")

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

# 1. Load and preprocess the dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# Normalize pixel values (0-255 -> 0-1)
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape to add the channel dimension CNNs expect: (28, 28) -> (28, 28, 1)
X_train_cnn = X_train.reshape(-1, 28, 28, 1)
X_test_cnn = X_test.reshape(-1, 28, 28, 1)

# 2. Display sample images
plt.figure(figsize=(12, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train[i], cmap="gray")
    plt.title(class_names[y_train[i]])
    plt.axis("off")
plt.tight_layout()
plt.savefig("sample_images.png")
plt.close()
print("Sample images saved as sample_images.png")

# 3. Build the CNN
model = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),

    # Convolution layer: detects local patterns (edges, textures) using
    # small filters slid across the image.
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu", name="conv_layer_1"),
    layers.MaxPooling2D(pool_size=(2, 2), name="pool_layer_1"),

    # A second convolution block lets the network build on the first layer's
    # simple features (edges) to detect more complex shapes.
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu", name="conv_layer_2"),
    layers.MaxPooling2D(pool_size=(2, 2), name="pool_layer_2"),

    layers.Flatten(name="flatten_layer"),
    layers.Dense(64, activation="relu", name="dense_layer"),
    layers.Dense(10, activation="softmax", name="output_layer"),
])

model.summary()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# 4. Train the CNN
print("\nTraining the CNN...")
history = model.fit(
    X_train_cnn, y_train,
    epochs=10,
    validation_split=0.2,
    verbose=2,
)

# 5. Evaluate on the test dataset
test_loss, test_accuracy = model.evaluate(X_test_cnn, y_test, verbose=0)
final_train_accuracy = history.history["accuracy"][-1]
final_val_accuracy = history.history["val_accuracy"][-1]

print(f"\nFinal Training Accuracy: {final_train_accuracy:.4f}")
print(f"Final Validation Accuracy: {final_val_accuracy:.4f}")
print(f"Final Test Accuracy: {test_accuracy:.4f}")
print(f"Final Test Loss: {test_loss:.4f}")

# 6. Predict the class of sample images, display predicted vs actual
predictions = model.predict(X_test_cnn, verbose=0)
predicted_labels = np.argmax(predictions, axis=1)

print("\nSample Predictions vs Actual Labels (first 8 test images):")
for i in range(8):
    print(f"Actual: {class_names[y_test[i]]:15} | Predicted: {class_names[predicted_labels[i]]}")

# 7. Plot training and validation accuracy AND loss curves
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

# 8. Confusion matrix
cm = confusion_matrix(y_test, predicted_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()
print("Confusion matrix saved as confusion_matrix.png")

# 9. Show 10 correctly classified and 10 incorrectly classified images
correct_indices = np.where(predicted_labels == y_test)[0]
incorrect_indices = np.where(predicted_labels != y_test)[0]

print(f"\nTotal correct predictions: {len(correct_indices)} / {len(y_test)}")
print(f"Total incorrect predictions: {len(incorrect_indices)} / {len(y_test)}")

# 10 correctly classified images
plt.figure(figsize=(14, 6))
for i, idx in enumerate(correct_indices[:10]):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[idx], cmap="gray")
    plt.title(f"Actual: {class_names[y_test[idx]]}\nPredicted: {class_names[predicted_labels[idx]]}",
              color="green", fontsize=9)
    plt.axis("off")
plt.suptitle("10 Correctly Classified Images")
plt.tight_layout()
plt.savefig("correctly_classified.png")
plt.close()
print("10 correctly classified images saved as correctly_classified.png")

# 10 incorrectly classified images
plt.figure(figsize=(14, 6))
for i, idx in enumerate(incorrect_indices[:10]):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[idx], cmap="gray")
    plt.title(f"Actual: {class_names[y_test[idx]]}\nPredicted: {class_names[predicted_labels[idx]]}",
              color="red", fontsize=9)
    plt.axis("off")
plt.suptitle("10 Incorrectly Classified Images")
plt.tight_layout()
plt.savefig("incorrectly_classified.png")
plt.close()
print("10 incorrectly classified images saved as incorrectly_classified.png")