# 🧬 Breast Tumor Cell Nuclei Segmentation using CNNs (U-Net + Transfer Learning Backbones)

![Python](https://img.shields.io/badge/Python-3.9-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-DeepLearning-red)
![OpenCV](https://img.shields.io/badge/OpenCV-ImageProcessing-lightgrey)
![Numpy](https://img.shields.io/badge/Numpy-Numerical-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![Project Type](https://img.shields.io/badge/Project-Segmentation-blueviolet)
![Category](https://img.shields.io/badge/Category-MedicalAI-critical)
![Deep Learning](https://img.shields.io/badge/DeepLearning-U--Net-important)
![Maintained](https://img.shields.io/badge/Maintained-yes-success)

![image](https://github.com/user-attachments/assets/4104b806-1c9d-41a1-af5b-99107385eaf9)

## 📌 Project Overview

This project focuses on the development of a **Convolutional Neural Network (CNN)** for **semantic segmentation of breast tumor cell nuclei** in histopathology images.

We implemented the **U-Net architecture** with various **pretrained encoder backbones**:

* VGG16
* ResNet50
* EfficientNetB0
* MobileNetV2
* U-Net Baseline (without transfer learning)

The main goal is to perform **pixel-wise segmentation**, accurately identifying cell nuclei regions to support digital pathology workflows.

## 🎯 Business Context

Breast cancer remains one of the most common and life-threatening cancers globally.
Accurate and early detection of **cell nuclei in breast tissue slides** is essential for:

* Assessing tumor aggressiveness
* Measuring mitotic activity
* Supporting clinical treatment decisions

> **Problem:** Manual segmentation is **subjective**, **time-consuming**, and prone to **human error**.

> **Objective:** Leverage **Deep Learning** to build an **automated segmentation tool** that generates **precise and consistent nuclei masks**, helping pathologists and reducing diagnosis time.

## 🏗️ Model Architectures Used

| Model                    | Encoder Backbone | Description                         |
| ------------------------ | ---------------- | ----------------------------------- |
| **U-Net (Baseline)**     | None             | Classical U-Net without pretraining |
| **VGG16-U-Net**          | VGG16 (ImageNet) | Deep CNN encoder                    |
| **ResNet50-U-Net**       | ResNet50         | Residual network                    |
| **EfficientNetB0-U-Net** | EfficientNetB0   | Lightweight efficient CNN           |
| **MobileNetV2-U-Net**    | MobileNetV2      | Mobile-optimized network            |

## 📈 Training and Validation Metrics

### 📉 Training Loss and Accuracy Curves

![image](https://github.com/user-attachments/assets/40bd643f-2769-4f0e-8474-ded2cb6d55f5)

### 🏅 Best Validation Accuracy Comparison

| Model          | Best Validation Loss | Best Validation Accuracy |
| -------------- | -------------------- | ------------------------ |
| **VGG16**      | **0.3153**           | **0.6848**               |
| MobileNetV2    | 0.3543               | 0.6758                   |
| U-Net          | 0.4027               | 0.6629                   |
| EfficientNetB0 | 0.5580               | 0.5795                   |
| ResNet50       | 0.6038               | 0.5674                   |

## 📈 ROC Curve (All Models)
![image](https://github.com/user-attachments/assets/f63e4f94-b022-478a-89b3-db4b5b8c706c)

## 📈 Training and validation loss and Training and Validation Accuracy
![image](https://github.com/user-attachments/assets/5f5f5d24-04b7-41a6-a555-aba252278fbd)



### ✅ Training Accuracy for Each Model

| Model          | Accuracy Curve                              |
| -------------- | ------------------------------------------- |
|U-Net| ![image](https://github.com/user-attachments/assets/d40f740f-e420-4dcd-97d7-08380a34b539)|
| VGG16          | ![VGG16 Accuracy](![image](https://github.com/user-attachments/assets/8833f86b-6a48-4f8d-b111-7c14c6cde4b1)|
| MobileNetV2    | ![image](https://github.com/user-attachments/assets/986427fc-2040-40b1-a714-e3d143258362)|
| EfficientNetB0 | ![image](https://github.com/user-attachments/assets/48cb0b4d-adbf-4dd4-9db6-e392bb34195d)|
| ResNet50       | ![image](https://github.com/user-attachments/assets/56e5ec22-8b40-4550-9af0-2e9b06208ec0)|

## 📊 ROC Curve and Precision-Recall (AUC and AP)

| ROC Curve                                    | Precision-Recall Curve                  |
| -------------------------------------------- | --------------------------------------- |
| ![image](https://github.com/user-attachments/assets/5e8c637e-66f5-4bee-a0a0-1bcf30efc3c3)|![image](https://github.com/user-attachments/assets/1f777e5d-908b-4f57-9ec3-9cec10f13949)|

## 🧪 Quantitative Metrics

| Metric                            | Formula                                    |
| --------------------------------- | ------------------------------------------ |
| **IoU (Intersection over Union)** | IoU = TP / (TP + FP + FN)                  |
| **Dice Coefficient**              | Dice = (2 × TP) / (2 × TP + FP + FN)       |
| **Pixel-wise Accuracy**           | Accuracy = (TP + TN) / (TP + TN + FP + FN) |
| **AUC (ROC Curve)**               | Area under the ROC Curve                   |
| **AP (Precision-Recall)**         | Average Precision                          |

## 🖼️ Qualitative Results – Visual Predictions

### ▶️ U-Net (Baseline):

![image](https://github.com/user-attachments/assets/c6c6b7d6-ae07-4c88-beb6-029fb19254f9)

### 🔎 U-Net (Baseline)

* ✅ Able to segment some nuclei areas.
* ⚠️ Poor boundary definition and several false positives and false negatives.
* 📉 Average IoU ranged between **0.3 and 0.55**.
* 🎯 Signs of **underfitting**, especially on dense nuclei regions


### ▶️ VGG16 + U-Net:

![image](https://github.com/user-attachments/assets/f5602809-c937-4fe4-8452-f653633b57f3)

* 🥇 **Best performing model** overall.
* ✅ Highly similar masks to ground truth, with **sharp boundaries and accurate nucleus coverage**.
* 📈 IoU between **0.65 and 0.74**, Dice between **0.78 and 0.85**.
* ✅ Excellent generalization for both sparse and dense nucleus regions.
* ❌ **Worst performance** across all models.
* ❌ Generated **completely blank masks (black images)**.
* 📉 IoU and Dice near **zero for all samples**.
* ❗ Needs hyperparameter tuning or different initialization strategies.


### ▶️ MobileNetV2 + U-Net:

![image](https://github.com/user-attachments/assets/19e72b2c-bd2e-402a-a267-6fae423aec6d)


### 🔎 MobileNetV2 + U-Net

* ✅ Lightweight model with **good overall performance**.
* ⚠️ Some **irregular edges** and occasional **over-segmentation**.
* 📉 IoU between **0.4 and 0.6**, Dice between **0.55 and 0.71**.
* ✅ Recommended for scenarios with hardware limitations.


## ✅ Summary and Conclusions

| Model          | Overall Performance | Visual Quality | Stability  |
| -------------- | ------------------- | -------------- | ---------- |
| VGG16          | 🥇 Excellent        | High           | Stable     |
| MobileNetV2    | ✅ Good              | Moderate       | Stable     |
| U-Net Baseline | ✅ Reasonable        | Moderate       | Acceptable |
| EfficientNetB0 | ⚠️ Poor             | Low            | Unstable   |
| ResNet50       | ❌ Very Poor         | None           | Failed     |

## 📈 Next Steps and Future Work

* Apply **advanced data augmentation** (elastic transformations, stain normalization).
* Experiment with **learning rate schedulers and early stopping**.
* Test **Transformer-based segmentation models (e.g., TransUNet)**.
* External validation using **larger and more diverse datasets**.
* Deploy as a **Streamlit Web App** for user-friendly inference.

## 📁 Project Directory Structure

```
├── data/                 # Raw dataset (images + masks)
├── notebooks/            # Jupyter Notebooks
├── models/               # Saved model weights
├── outputs/              # Result metrics and images
├── utils/                # Custom utility functions
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## 🛠️ Technologies Used

* Python 3.9
* TensorFlow / Keras
* NumPy
* OpenCV
* Matplotlib
* Scikit-learn
* Jupyter Notebooks

## 📚 References

* Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*
* Kaggle TNBC Nuclei Dataset
* TensorFlow Hub – Pretrained CNN Models
* ImageNet Dataset for Backbone Pretraining
