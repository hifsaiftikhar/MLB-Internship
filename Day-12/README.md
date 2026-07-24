# Day-12: Transfer Learning and Cats vs Dogs Image Classifier

## What I worked on

Today's focus was Transfer Learning - using a pre-trained CNN (MobileNetV2) as a feature-extracting backbone, rather than training a network from scratch, to classify images of cats and dogs.

## What is Transfer Learning?

Transfer Learning reuses a model that has already been trained on a large dataset (in this case, MobileNetV2 trained on millions of ImageNet images) and adapts it to a new, related task. Instead of learning to recognize edges, textures, and shapes from scratch, I start from a model that already knows how to do this well, and only train a small classification head on top of it for my specific task (cats vs dogs). This is why Transfer Learning is so widely used in industry: it needs far less data and training time than training a CNN from scratch, while often producing better results.

**Feature Extraction** (what I used today): freeze the pre-trained base model entirely and only train the new classification layers added on top.

**Fine-Tuning** (not used today, but a natural next step): unfreeze some of the later layers of the pre-trained base and train them at a very low learning rate, so the model can adjust its high-level features slightly for the new task, while keeping the early, more general layers frozen.

## Why I chose MobileNetV2

MobileNetV2 is a lightweight CNN architecture designed to run efficiently on mobile and edge devices, while still achieving strong accuracy. Compared to larger architectures like VGG16 or ResNet50, MobileNetV2 has far fewer parameters and runs faster, which matters when training and experimenting on a personal CPU rather than dedicated GPU hardware. Since this project's task (binary cat vs dog classification) doesn't need the extra capacity of a much larger model, MobileNetV2 was a practical choice - it kept training time reasonable while still comfortably clearing the accuracy target.

## Model architecture

MobileNetV2 (frozen, include_top=False) -> GlobalAveragePooling2D -> Dense(128, ReLU) -> Dropout(0.2) -> Dense(1, Sigmoid)

- The frozen MobileNetV2 base acts as a fixed feature extractor (2,257,984 parameters, none of them trainable).
- GlobalAveragePooling2D condenses each feature map down to a single number per channel, before the final classification layers.
- Dropout(0.2) randomly disables 20% of neurons during training, which helps reduce overfitting.
- The output layer uses Sigmoid for binary classification (cat vs dog).
- Total parameters: 2,422,081, of which only 164,097 are actually trainable - the rest belong to the frozen base.

## Final validation accuracy

- **Final Training Accuracy: 99.69%**
- **Final Validation Accuracy: 98.29%**
- Final Validation Loss: 0.0463

This clears both the minimum target (90%) and the stretch target (93%+) by a wide margin, on the very first full training run, without needing any fine-tuning or additional experimentation.

## Experiments performed

Since the first run already exceeded the 93% target by a comfortable margin, I did not need to run additional experiments (adjusting epochs, learning rate, batch size, or fine-tuning) to hit the performance goal. Given how strong the result already was, the frozen MobileNetV2 backbone combined with a simple classification head was sufficient - this reflects how much useful, general-purpose visual knowledge a model pre-trained on millions of images already carries into a new but related task.

## Key challenges and lessons learned

**A Windows-specific bug in tensorflow_datasets:** the task suggested loading the dataset with `tfds.load("cats_vs_dogs", ...)`. On my machine, this repeatedly failed with a `KeyError` while extracting images from the dataset's underlying zip file. After investigating, I found this is a known bug specific to Windows: zip archives always use forward slashes internally, but tensorflow_datasets builds its internal file lookup path using Windows-style backslashes, so it fails to find images that are genuinely present in the archive. This does not happen on Mac/Linux.

To work around this, I switched to downloading the dataset directly (the same zip file, sourced directly from Microsoft) and loading it with `tf.keras.utils.image_dataset_from_directory` instead of `tfds.load` - this is the same approach used in TensorFlow's own official "Image classification from scratch" tutorial, and it avoids the buggy zip-parsing path entirely.

**Corrupted images in the dataset:** this real-world dataset contains a small number of files that are not valid JPEGs (missing the expected JFIF header). Loading these without filtering would crash training partway through. I added a check that scans every image, testing for the JFIF marker, and removes any that fail - this removed about 1,580-1,590 corrupted images out of roughly 25,000 total, leaving 23,410-23,422 valid images to work with.

**Training time on CPU vs GPU:** the first training attempt on my local CPU was still on epoch 1 after 8 minutes. I moved the same script to Google Colab with a free GPU, where all 10 epochs completed in about 6 minutes total - a clear, practical illustration of why GPU acceleration matters for CNN-based transfer learning, even with a frozen, lightweight backbone like MobileNetV2.

## Files

- transfer_learning_practice.py: Practice 1 (load/explore/freeze MobileNetV2, add a classification head) and Practice 2 (load, filter, resize, and split the Cats vs Dogs dataset)
- cats_vs_dogs_classifier.py: full mini project - transfer learning pipeline, training, evaluation, and predictions
- training_accuracy_loss.png: training vs validation accuracy and loss curves across 10 epochs
- sample_predictions.png: 8 sample test images with actual vs predicted labels
- README.md: this file

## Author

Hifsa Iftikhar
