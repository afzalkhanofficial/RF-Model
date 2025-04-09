# 🛢️ Oil Spill Detection using Random Forest

This project implements a machine learning pipeline to classify satellite images into **oil spill** and **non-oil spill** categories. It leverages data augmentation, feature extraction (HOG and color histograms), and a Random Forest classifier.

## 📦 Technologies Used
- Python 3.x
- NumPy, OpenCV, Matplotlib
- scikit-image for HOG features
- scikit-learn for ML and model evaluation
- TensorFlow Keras for data augmentation
- Pillow for image processing
- Joblib for model persistence

## 📁 Dataset Structure
```
RF-Model/
├── requirements.txt
├── app.py
├── oil_spill_rf_model.pkl
└── feature_scaler.pkl
```

## 🚀 Features
- Image preprocessing and augmentation using Keras
- HOG feature extraction from grayscale images
- Color histogram computation (RGB, 32 bins per channel)
- Random Forest classifier with GridSearchCV hyperparameter tuning
- Model and scaler persistence via Joblib

## 🛠️ Requirements
Install dependencies:
```bash
pip install numpy opencv-python scikit-image matplotlib Pillow scikit-learn tensorflow joblib
```

## 🧠 How it Works
1. **Preprocess** images to a standard size of `128x128`.
2. **Augment** training images using transformations.
3. **Extract features**:
   - HOG (Histogram of Oriented Gradients) from grayscale images
   - Color histograms (RGB, normalized)
4. **Train** a Random Forest model using `GridSearchCV`.
5. **Evaluate** with accuracy and classification report.
6. **Save** the model (`oil_spill_rf_model.pkl`) and the scaler (`feature_scaler.pkl`).

## 👤 Author
**Afzal Khan**

---

Feel free to fork this project, customize the pipeline, or adapt it to new datasets!
