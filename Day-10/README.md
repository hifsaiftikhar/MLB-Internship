# Day-10: Introduction to Deep Learning

## What I worked on

Today marked the start of Phase 2 (Deep Learning). I learned the fundamentals of neural networks, built simple Artificial Neural Networks (ANNs) using TensorFlow/Keras, experimented with different activation functions, and trained a full ANN on the Fashion MNIST dataset.

## What is Deep Learning?

Deep Learning is a subset of Machine Learning based on Artificial Neural Networks with multiple layers ("deep" refers to having many layers between input and output). Instead of relying on manually engineered features, a deep network learns to extract relevant patterns directly from raw data (like pixel values in an image) through its layers.

## Difference between Machine Learning and Deep Learning

Traditional Machine Learning models (like the Logistic Regression and Linear Regression models from earlier days) typically work well with structured, tabular data and often need manually selected or engineered features. Deep Learning models, especially neural networks, can automatically learn useful features directly from raw, high-dimensional data such as images or text, but generally need much more data and computation to do so effectively. For a small structured dataset like the Breast Cancer dataset from Day-8, a simple model like Logistic Regression is often more practical than a deep network; for something like image classification (today's Fashion MNIST task), a neural network is far better suited.

## What is a Perceptron?

A Perceptron is the simplest unit of a neural network - a single artificial neuron. It takes in one or more inputs, multiplies each by a weight, sums them together with a bias term, and passes the result through an activation function to produce an output. A single Perceptron can only make simple linear decisions (like a basic yes/no classifier); stacking many of them together into layers, with non-linear activation functions, is what allows a full neural network to learn much more complex patterns.

## Activation functions explored

- **ReLU (Rectified Linear Unit):** outputs the input directly if it's positive, otherwise outputs 0. It's fast to compute and is the most commonly used activation function in hidden layers of modern networks, including the hidden layer in this project's Fashion MNIST model.
- **Sigmoid:** squashes any input into a range between 0 and 1. Commonly used in the output layer for binary classification problems, where the output represents a probability of belonging to one of two classes.
- **Tanh:** squashes any input into a range between -1 and 1. Similar to Sigmoid but centered at 0, which can help training converge faster than Sigmoid when used in hidden layers.
- **Softmax** (used in the Fashion MNIST project's output layer): converts a set of raw outputs into probabilities across multiple classes that all sum to 1 - used for multi-class classification, where Sigmoid alone isn't suitable since there are more than 2 possible categories.

In activation_functions.py, I built the same network architecture three times, changing only the hidden layer's activation function (ReLU, Sigmoid, Tanh) each time. The model summaries confirmed that changing the activation function does not change the model's structure or parameter count - it only changes how each neuron transforms its input.

## Coding practice

- **tensorflow_verification.py:** verified TensorFlow and Keras were installed correctly, and ran a basic tensor operation to confirm everything works.
- **simple_ann.py:** built a simple ANN with one Input Layer, one Hidden Layer (16 neurons, ReLU), and one Output Layer (1 neuron, Sigmoid), and printed the model summary with an explanation of each layer's role.
- **activation_functions.py:** compared ReLU, Sigmoid, and Tanh in the hidden layer of the same architecture.

## Mini Project: Fashion MNIST ANN (fashion_mnist_ann.py)

- Loaded the Fashion MNIST dataset directly from tensorflow.keras.datasets (60,000 training images, 10,000 test images, each 28x28 grayscale pixels across 10 clothing categories).
- Explored the dataset's shape and displayed sample training images with their labels (sample_images.png).
- Normalized pixel values from the original 0-255 range down to 0-1, which helps the network train faster and more reliably.
- Built an ANN with a Flatten layer (converts each 28x28 image into a 784-value vector), a Hidden Layer (128 neurons, ReLU), and an Output Layer (10 neurons, Softmax, one per clothing category).
- Trained for 10 epochs, holding out 20% of the training data for validation.
- Evaluated on the test dataset and plotted training vs validation accuracy across epochs (training_accuracy_curve.png).
- Made predictions on 8 test images and displayed the predicted label against the actual label, both as text output and as an image grid with color-coded correct/incorrect predictions (sample_predictions.png).

### Model's final training and testing accuracy

- Final Training Accuracy: 90.88%
- Final Validation Accuracy: 88.24%
- Final Test Accuracy: 87.47%

Training accuracy ended up a bit higher than validation and test accuracy, which is a normal and expected pattern - the model fits the training data slightly better than data it hasn't directly trained on, but the gap here is small, so this isn't a sign of serious overfitting.

## Challenges faced

- The Fashion MNIST dataset needs to be downloaded from Google's servers the first time load_data() is called, so a stable internet connection is required on first run; after that, it's cached locally.
- Deciding on the number of neurons and epochs for the ANN involved some judgment rather than a fixed rule - 128 neurons and 10 epochs gave a reasonable balance between training time and accuracy for this dataset, without the training/validation gap growing large enough to suggest overfitting.

## Files

- tensorflow_verification.py: TensorFlow/Keras installation check
- simple_ann.py: simple ANN with input, hidden, and output layer, with model summary
- activation_functions.py: ReLU vs Sigmoid vs Tanh comparison
- fashion_mnist_ann.py: full mini project - Fashion MNIST ANN
- sample_images.png: sample training images with labels
- training_accuracy_curve.png: training vs validation accuracy over epochs
- sample_predictions.png: sample test images with actual vs predicted labels
- README.md: this file

## Author

Hifsa Iftikhar