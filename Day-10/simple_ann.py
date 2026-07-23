from tensorflow import keras
from tensorflow.keras import layers

# Build a simple Artificial Neural Network with:
# - One Input Layer
# - One Hidden Layer
# - One Output Layer

model = keras.Sequential([
    # Input Layer: expects data with 20 features per sample.
    keras.Input(shape=(20,)),

    # Hidden Layer: 16 neurons, using ReLU activation.
    layers.Dense(16, activation="relu", name="hidden_layer"),

    # Output Layer: 1 neuron, using Sigmoid activation.
    layers.Dense(1, activation="sigmoid", name="output_layer"),
])

# Print the model summary - shows each layer, its output shape, and how many
# trainable parameters (weights and biases) it has.
model.summary()

print("\nLayer-by-layer explanation:")
print("Input Layer: accepts 20 input features per sample, no computation happens here.")
print("Hidden Layer: 16 neurons with ReLU activation, learns patterns from the input.")
print("Output Layer: 1 neuron with Sigmoid activation, outputs a value between 0 and 1")
print("(suitable for a binary classification problem).")