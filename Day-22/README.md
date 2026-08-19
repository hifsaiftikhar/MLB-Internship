
# Day-22: Document & Object Segmentation Tool

##  Overview

This project focuses on the fundamentals of **Image Segmentation** using Python and OpenCV.

The project applies different thresholding techniques to separate foreground objects from the background. A Gradio-based application is also developed to allow users to upload an image and compare the segmentation results interactively.

##  Objectives

* Understand the basics of image segmentation.
* Convert images to grayscale.
* Apply Binary Thresholding.
* Apply Adaptive Thresholding.
* Apply Otsu Thresholding.
* Perform simple foreground/background segmentation.
* Compare the results of different thresholding methods.
* Build an interactive Gradio application.

##  What is Image Segmentation?

Image segmentation is a computer vision technique used to divide an image into meaningful regions or groups of pixels.

Unlike object detection, which identifies objects using bounding boxes, segmentation works at the **pixel level** and attempts to separate the foreground from the background.

Image segmentation is commonly used in:

* Medical image analysis
* Autonomous vehicles
* Agriculture
* Document processing
* Object extraction
* Image editing

## 🔍 Thresholding Techniques

### 1. Binary Thresholding

Binary thresholding uses a fixed threshold value to divide pixels into two groups.

In this project, a threshold value of **127** is used.

Pixels above the threshold are converted to white, while pixels below the threshold are converted to black.

**Advantages:**

* Simple and fast
* Easy to implement
* Works well when the lighting is consistent

**Limitation:**

* May not perform well with uneven lighting.

### 2. Adaptive Thresholding

Adaptive thresholding calculates a different threshold for different regions of an image.

This makes it more suitable for images with uneven lighting or shadows.

In this project, **Gaussian adaptive thresholding** is used with a block size of 11.

**Advantages:**

* Handles uneven lighting better
* Works well for documents and local variations

**Limitation:**

* Can produce more noise in some images.

### 3. Otsu Thresholding

Otsu thresholding automatically calculates an optimal threshold value based on the image histogram.

It is useful when the foreground and background have relatively different intensity distributions.

**Advantages:**

* Automatically determines the threshold
* No need to manually select a threshold value
* Simple and effective for many images

**Limitation:**

* Performance can decrease with complex backgrounds or strong lighting variations.

##  Method Comparison

| Method   | Threshold   | Best Used For                                      |
| -------- | ----------- | -------------------------------------------------- |
| Binary   | Fixed (127) | Simple images with consistent lighting             |
| Adaptive | Local       | Uneven lighting and shadows                        |
| Otsu     | Automatic   | Images with clear foreground/background separation |

##  Dataset

The project uses sample images for testing different segmentation techniques.

The dataset contains images with different objects, scenes, textures, and lighting conditions.

The images are stored in the `input/` folder.

### Dataset Structure

```text
input/
├── sample01.jpg
├── sample02.jpg
├── sample03.png
├── sample04.png
├── ...
└── sample15.jpg
```

Some images were obtained from publicly available OpenCV sample data, while additional images can be added to test documents, shadows, and uneven lighting.

##  Image Processing Pipeline

The processing pipeline follows these steps:

```text
Input Image
     ↓
Grayscale Conversion
     ↓
 ┌───────────────┬──────────────────┬───────────────┐
 ↓               ↓                  ↓
Binary        Adaptive             Otsu
Threshold     Threshold          Threshold
 ↓               ↓                  ↓
 └───────────────┴──────────────────┴───────────────┘
                     ↓
          Foreground Segmentation
                     ↓
          Morphological Processing
                     ↓
             Final Output
```

##  Technologies Used

* Python
* OpenCV
* NumPy
* Gradio

##  Project Structure

```text
Day-22/
│
├── input/
│   ├── sample01.jpg
│   ├── sample02.jpg
│   └── ...
│
├── output/
│   ├── sample01_1_binary.png
│   ├── sample01_2_adaptive.png
│   ├── sample01_3_otsu.png
│   ├── sample01_4_foreground.png
│   └── ...
│
├── download_images.py
├── segmentation.py
├── app.py
├── requirements.txt
└── README.md
```

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
```

### 2. Navigate to the Project

```bash
cd Day-22
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Prepare the Dataset

Run:

```bash
python download_images.py
```

The sample images will be downloaded into the `input/` folder.

### 5. Run Image Segmentation

```bash
python segmentation.py
```

The processed results will be saved in the `output/` folder.

### 6. Run the Gradio Application

```bash
python app.py
```

The application provides a local Gradio interface where users can upload an image or select one of the available sample images.

##  Gradio Application

The application provides:

* Image upload functionality
* Sample images for testing
* Binary Thresholding
* Adaptive Thresholding
* Otsu Thresholding
* Foreground/background segmentation
* Automatic saving of the best segmentation result

##  Best Performing Method

For this dataset, **Otsu Thresholding combined with morphological processing** provided the most useful overall foreground segmentation results.

Otsu automatically determines the threshold value, while morphological operations help reduce small noise and improve the resulting foreground mask.

However, the best method can vary depending on the image. Adaptive Thresholding can perform better on images with uneven lighting or shadows, while Binary Thresholding works well for simpler images with consistent lighting.

##  Challenges Faced

During implementation, some challenges included:

* Different images have different brightness and contrast levels.
* A fixed threshold does not work equally well for every image.
* Uneven lighting can affect global thresholding methods.
* Shadows and image noise can produce unwanted regions.
* Morphological operations were required to improve the foreground mask.
* Different image types and sizes required consistent preprocessing.

##  Output

For every input image, the script generates four outputs:

```text
Binary Threshold
Adaptive Threshold
Otsu Threshold
Foreground Segmentation
```

The outputs are stored in the `output/` folder for comparison.

## Author

**Hifsa Iftikhar**
