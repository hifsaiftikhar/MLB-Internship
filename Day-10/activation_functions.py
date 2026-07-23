from tensorflow import keras
from tensorflow.keras import layers

# Build the same simple network architecture three times, changing only the
# hidden layer's activation function each time, to observe how the choice of
# activation function affects the model.

activations_to_try = ["relu", "sigmoid", "tanh"]

for activation in activations_to_try:
    print(f"\n===== Hidden Layer Activation: {activation.upper()} =====")

    model = keras.Sequential([
        keras.Input(shape=(20,)),
        layers.Dense(16, activation=activation, name="hidden_layer"),
        layers.Dense(1, activation="sigmoid", name="output_layer"),
    ])

    model.summary()

print("\n===== Observations =====")
print("Changing the activation function does NOT change the model's structure")
print("(same number of layers, same number of neurons, same parameter count).")
print("It only changes HOW each neuron transforms its input before passing it on:")
print()
print("ReLU: outputs the input directly if positive, otherwise outputs 0.")
print("      Fast to compute, commonly used in hidden layers of most modern networks.")
print()
print("Sigmoid: squashes any input into a range between 0 and 1.")
print("         Useful for binary classification outputs, but can slow down")
print("         learning in deep hidden layers (vanishing gradient problem).")
print()
print("Tanh: squashes any input into a range between -1 and 1.")
print("      Similar to Sigmoid but centered at 0, which can help training")
print("      converge faster than Sigmoid in hidden layers.")