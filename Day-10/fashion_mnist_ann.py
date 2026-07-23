import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import fashion_mnist

print("===== FASHION MNIST - ARTIFICIAL NEURAL NETWORK =====\n")

# Class names for Fashion MNIST (the dataset only stores numeric labels 0-9)
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

# 1. Load the Fashion MNIST dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# 2. Explore the dataset
print("Training data shape:", X_train.shape)
print("Training labels shape:", y_train.shape)
print("Test data shape:", X_test.shape)
print("Test labels shape:", y_test.shape)
print("\nPixel value range before normalization:", X_train.min(), "to", X_train.max())
print("Sample label:", y_train[0], "->", class_names[y_train[0]])

# Show a few sample images from the training set
plt.figure(figsize=(10, 4))
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(X_train[i], cmap="gray")
    plt.title(class_names[y_train[i]])
    plt.axis("off")
plt.tight_layout()
plt.savefig("sample_images.png")
plt.close()
print("\nSample training images saved as sample_images.png")

# 3. Normalize the pixel values from the 0-255 range down to 0-1.
# This helps the network train faster and more reliably, since large raw
# pixel values can make gradients unstable during training.
X_train = X_train / 255.0
X_test = X_test / 255.0
print("\nPixel value range after normalization:", X_train.min(), "to", X_train.max())

# 4. Build a simple Artificial Neural Network
model = keras.Sequential([
    # Flatten turns each 28x28 image into a single 784-value vector,
    # since a basic Dense (fully connected) layer expects 1D input per sample.
    keras.Input(shape=(28, 28)),
    layers.Flatten(),

    # Hidden layer: learns patterns from the flattened pixel values.
    layers.Dense(128, activation="relu", name="hidden_layer"),

    # Output layer: 10 neurons (one per clothing category), using Softmax,
    # which converts the outputs into probabilities across all 10 classes
    # that sum to 1 - the class with the highest probability is the prediction.
    layers.Dense(10, activation="softmax", name="output_layer"),
])

model.summary()

# 5. Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# 6. Train the model, holding out 20% of the training data for validation
# (data the model doesn't train on, used to check it isn't just memorizing)
print("\nTraining the model...")
history = model.fit(
    X_train, y_train,
    epochs=10,
    validation_split=0.2,
    verbose=2,
)

# 7. Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")
print(f"Final Test Loss: {test_loss:.4f}")

final_train_accuracy = history.history["accuracy"][-1]
final_val_accuracy = history.history["val_accuracy"][-1]
print(f"Final Training Accuracy: {final_train_accuracy:.4f}")
print(f"Final Validation Accuracy: {final_val_accuracy:.4f}")

# 8. Plot training and validation accuracy curves (bonus)
plt.figure(figsize=(8, 6))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("training_accuracy_curve.png")
plt.close()
print("\nAccuracy curve saved as training_accuracy_curve.png")

# 9. Make predictions on a few test images
predictions = model.predict(X_test[:8], verbose=0)
predicted_labels = np.argmax(predictions, axis=1)

print("\nSample Predictions vs Actual Labels:")
for i in range(8):
    print(f"Actual: {class_names[y_test[i]]:15} | Predicted: {class_names[predicted_labels[i]]}")

# 10. Display sample predictions with their corresponding images (bonus)
plt.figure(figsize=(12, 5))
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(X_test[i], cmap="gray")
    color = "green" if predicted_labels[i] == y_test[i] else "red"
    plt.title(f"Actual: {class_names[y_test[i]]}\nPredicted: {class_names[predicted_labels[i]]}",
              color=color, fontsize=9)
    plt.axis("off")
plt.tight_layout()
plt.savefig("sample_predictions.png")
plt.close()
print("\nSample predictions with images saved as sample_predictions.png")