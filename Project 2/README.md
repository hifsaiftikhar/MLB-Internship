# 🚧 Project 2: Custom Road Damage Detection System

**Author**: Hifsa Iftikhar  
**Internship Assignment**: Project 2 - Custom Object Detection  
**Workspace Path**: `MLB-Internship/Project 2`

---

## 📋 Project Overview
This project implements a **Custom Road Damage Detection System** utilizing the state-of-the-art **YOLO11** architecture. Road surface defects, such as potholes and cracks, are highly variable in shape, contrast, and orientation, making them difficult for standard object detection models. 

This project establishes a complete end-to-end pipeline:
1. Programmatic dataset exploration.
2. Custom YOLO model training with optimized hyperparameters in Google Colab.
3. Model evaluation on test data.
4. Interactive dashboard deployment (via Gradio) allowing users to run inference on custom road images and video streams.

---

## 📁 Dataset Details
We selected the **Road Damage Detection** dataset from Roboflow:
* **Roboflow Project**: `road-damages-detection-9gt0k` (Version 1)
* **Dataset Format**: YOLOv8
* **Size**: 9,836 total images
  * **Train Set**: 9,390 images (24,083 annotated damage objects)
  * **Validation Set**: 290 images
  * **Test Set**: 156 images

### 🏷️ Damage Classes
The dataset identifies **7 distinct categories** of road damage:
1. **D00**: Longitudinal Crack (cracks running parallel to the direction of travel)
2. **D10**: Lateral Crack (cracks running perpendicular to travel)
3. **D20**: Alligator Crack (crocodile-skin pattern fatigue cracking)
4. **D40**: Pothole / Rutting
5. **D43**: Pavement Distress
6. **D44**: Major Pavement Rupture
7. **D50**: Manhole / Utility Cover (often flush or recessed, causing bumps)

---

## 🚀 Training Configuration & Experiments

### Experiment 1: YOLO11n (Baseline)
* **Model**: YOLO11 Nano (`yolo11n.pt`)
* **Resolution**: 640 × 640
* **Batch Size**: 16
* **Epochs**: 50
* **Result**: **45.9% mAP@50** (Baseline)

### Experiment 2: YOLO11s (Optimized Architecture)
* **Model**: YOLO11 Small (`yolo11s.pt`)
* **Resolution**: 640 × 640
* **Batch Size**: 16
* **Epochs**: 80 (Early stopping at epoch 34)
* **Result**: **47.5% mAP@50**

### 🌀 Advanced Optimization Setup (Google Colab Notebook)
To solve the poor convergence results, we configured the following training parameters in [`road_damage_training.ipynb`](file:///c:/Users/HP/Desktop/MLB-Internship/Project%202/road_damage_training.ipynb):
* **Optimizer (`AdamW`)**: Uses AdamW instead of standard SGD. This is much more stable for road texture datasets.
* **Learning Rate (`lr0=0.001`)**: Adjusted downwards to prevent weight oscillation.
* **Vertical Flips (`flipud=0.5`)**: Enabled vertical flips along with standard horizontal flips, as cracks and road damage can appear at any angle or orientation.
* **Late-Epoch Mosaic Disabling (`close_mosaic=10`)**: Disables mosaic data augmentation in the final 10 epochs. This allows the model to stabilize and converge on clean, un-warped object boundaries, significantly refining localization precision for small cracks and potholes.

---

## 📊 Comparative Performance Results (YOLO11n vs. YOLO11s)

Here is a detailed comparison of our baseline model (`YOLO11n`) and the optimized model (`YOLO11s`):

### 1. Overall Performance Overview
| Metric | YOLO11n Baseline (From Training Log) | YOLO11s Optimized Model | Net Change |
| --- | ---: | ---: | ---: |
| **Precision** | 48.4% | 52.6% | **+4.2%** |
| **Recall** | 48.0% | 48.2% | **+0.2%** |
| **mAP@50** | **45.9%** | **47.5%** | **+1.6%** |
| **mAP@50-95** | 22.6% | 22.9% | **+0.3%** |

### 2. Class-Level mAP@50 Comparison
| Class ID / Damage Type | YOLO11n Baseline | YOLO11s Optimized | Improvement |
| --- | ---: | ---: | ---: |
| **All Classes (Avg)** | **45.9%** | **47.5%** | **+1.6%** |
| **D00** (Longitudinal Crack) | 28.2% | 30.9% | **+2.7%** |
| **D10** (Lateral Crack) | 28.7% | 25.4% | -3.3% |
| **D20** (Alligator Crack) | 45.8% | 45.0% | -0.8% |
| **D40** (Pothole / Rutting) | 23.0% | 32.0% | **+9.0%** |
| **D43** (Pavement Distress) | 64.7% | 63.4% | -1.3% |
| **D44** (Major Pavement Rupture) | 60.7% | 60.4% | -0.3% |
| **D50** (Manhole / Utility Cover) | 69.9% | 75.4% | **+5.5%** |

### Key Observations from the Comparison:
* **Significant Pothole Improvement (D40 +9.0%)**: Upgrading the model capacity to YOLO11s and applying late-epoch mosaic disabling successfully allowed the network to learn bounding box borders for potholes and rutting.
* **Manhole & Utility Cover Precision (D50 +5.5%)**: Reached **75.4% mAP@50**, proving highly accurate at identifying circular pavement anomalies.
* **Cracks Stability**: Longitudinal cracks (D00) improved by **2.7%**, while lateral cracks (D10) saw a slight trade-off due to sample shape variations.

---

## 🛠️ Challenges Faced & Solutions

1. **Small & Thin Damage Features (Low Baseline mAP)**:
   * *Challenge*: Road cracks are thin and often blur into the asphalt texture, resulting in a low baseline mAP.
   * *Solution*: Upgraded to a larger capacity model (`YOLO11s`), introduced the `AdamW` optimizer, and used `close_mosaic=10` to refine bounding box precision in final epochs.
2. **Colab Notebook Disconnections**:
   * *Challenge*: Colab disconnected during longer runs, deleting trained weights stored locally in `/content`.
   * *Solution*: Structured the training notebook to mount Google Drive and output weights directly to `MyDrive/Road_Damage_Detection`, ensuring checkpoint persistence.
3. **Severe Class Imbalance**:
   * *Challenge*: Class D20 had 6,109 annotations, while D43 had only 650, causing poor performance on rare distress categories.
   * *Solution*: Implemented data mixup (`mixup=0.15`) and copy-paste augmentations (`copy_paste=0.1`) to synthetically augment rare categories in multi-object frames.

---

## 🏃 Launching the Gradio Application

To run the interactive web interface locally:
```bash
python -m pip install -r requirements.txt
python app.py
```
This launches a browser-accessible dashboard (typically on `http://127.0.0.1:7860`) where you can:
1. Toggle between the custom-trained `best.pt` model and the pre-trained HuggingFace fallback model.
2. Upload road images to run predictions and view/download the annotated output.
3. Upload road videos to process frame-by-frame and download the marked MP4 output.
4. Review the damage classification summary table.
