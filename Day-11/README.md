# Day-11: Convolutional Neural Networks (CNN) and Image Classification

## What I worked on

Today's focus was Convolutional Neural Networks (CNNs) - the architecture behind most modern Computer Vision systems. I built and trained CNNs on the Fashion MNIST dataset, compared the result against yesterday's plain ANN, and evaluated performance using accuracy/loss curves, a confusion matrix, and correctly/incorrectly classified sample images.

## Why CNNs are better than ANNs for image data

A plain ANN (like the one from Day-10) has to flatten an image into a single long list of pixel values before it can process it, which means it loses all information about which pixels are spatially near each other. Two pixels that are right next to each other in the real image end up just as "far apart" to the network as two pixels on opposite corners.

A CNN instead uses convolution layers that slide small filters directly across the 2D image, so it can detect local patterns - edges, curves, textures - based on how nearby pixels relate to each other. This makes CNNs far better suited to image data, since spatial structure (what's next to what) is exactly the kind of information that matters most in images.

This showed up directly in my results: the same Fashion MNIST dataset reached about 87.5% test accuracy with a plain ANN yesterday, but about 90.4% test accuracy with a CNN today - a real, measurable improvement from the same data, just processed with a spatially-aware architecture.

## The purpose of convolution and pooling layers

- **Convolution layer:** slides a small filter (kernel), for example 3x3 pixels, across the image, computing a weighted sum at each position. Each filter learns to detect a specific pattern - early layers typically learn simple things like edges, and with more layers, the network can combine these into more complex shapes. The output of a convolution layer is called a feature map.
- **Max Pooling layer:** shrinks each feature map by keeping only the maximum value in each small region (for example, each 2x2 block). This reduces the amount of data the network has to process going forward, while keeping the strongest detected features, and also makes the network somewhat more tolerant to small shifts in where a feature appears in the image.
- **Flatten layer:** once convolution and pooling have extracted spatial features, Flatten converts the resulting 2D feature maps into a single 1D vector, since the final Dense (fully connected) layers need 1D input.
- **Dense (fully connected) layer:** combines the extracted features to make the actual classification decision.

## My model architecture

**cnn_practice.py** (Practice 2 and 3): a single-block CNN -
Conv2D (32 filters) -> MaxPooling2D -> Flatten -> Dense (64, ReLU) -> Output (10, Softmax)

**fashion_mnist_cnn_classifier.py** (mini project): a two-block CNN -
Conv2D (32 filters) -> MaxPooling2D -> Conv2D (64 filters) -> MaxPooling2D -> Flatten -> Dense (64, ReLU) -> Output (10, Softmax)

The second convolution block lets the network build on the simple features detected by the first block (like edges) to recognize more complex shapes, which is part of why the mini project's CNN performed slightly better than the single-block practice version.

## Final training and testing accuracy

**cnn_practice.py** (single conv block, 5 epochs):
- Final Training Accuracy: 92.78%
- Final Test Accuracy: 90.33%

**fashion_mnist_cnn_classifier.py** (two conv blocks, 10 epochs):
- Final Training Accuracy: 94.77%
- Final Validation Accuracy: 90.90%
- Final Test Accuracy: 90.43%
- Final Test Loss: 0.2898

Graphs and confusion matrix (see training_accuracy_loss.png and confusion_matrix.png):
- The accuracy/loss curves show both training and validation accuracy improving steadily across the 10 epochs, with validation accuracy leveling off around epoch 7-8 while training accuracy kept climbing slightly further - a mild, expected gap rather than a sign of serious overfitting.
- The confusion matrix shows the model does very well on visually distinct classes (Trouser, Bag, Ankle boot), but confuses visually similar classes more often - particularly Shirt vs T-shirt/top and Shirt vs Pullover/Coat, which makes sense since these categories can look quite similar in a low-resolution grayscale image.

Out of 10,000 test images, 9,043 were classified correctly and 957 incorrectly (see correctly_classified.png and incorrectly_classified.png for sample images of each).

## Challenges faced

- CNNs expect image input with an explicit channel dimension (height, width, channels), but Fashion MNIST images are loaded as plain (28, 28) grayscale arrays. This needed an explicit reshape to (28, 28, 1) before the data could be passed into the Conv2D layers.
- Training a CNN is noticeably slower than the plain ANN from yesterday, since convolution operations are more computationally expensive per epoch than simple Dense layers - each epoch here took roughly 13-14 seconds on CPU, compared to about 3-5 seconds for yesterday's ANN.
- Interpreting the confusion matrix meaningfully took more than just reading off the highest numbers - the misclassifications cluster around genuinely visually-similar clothing categories, which is a more useful and specific insight than just reporting overall accuracy.

## Files

- cnn_practice.py: Practice 1-3 - load/visualize/normalize the dataset, build and train a simple CNN, evaluate it
- practice_sample_images.png: 10 sample training images with labels (from cnn_practice.py)
- fashion_mnist_cnn_classifier.py: full mini project - two-block CNN, training, evaluation, confusion matrix, correct/incorrect predictions
- sample_images.png: sample training images with labels (from the mini project)
- training_accuracy_loss.png: training vs validation accuracy and loss curves
- confusion_matrix.png: confusion matrix across all 10 clothing categories
- correctly_classified.png: 10 correctly classified test images
- incorrectly_classified.png: 10 incorrectly classified test images
- README.md: this file

## Author

Hifsa Iftikhar