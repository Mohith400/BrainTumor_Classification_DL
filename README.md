# 🧠 Brain Tumor Detection using CNN

## 📊 Overview

This project implements a **Convolutional Neural Network (CNN)** using **TensorFlow/Keras** to classify brain MRI images as either:

* **No Tumor** 🟢
* **Tumor Present** 🔴

The dataset is divided into two folders:

* `datasets/no/` → MRI scans without tumor
* `datasets/yes/` → MRI scans with tumor

The model is trained on preprocessed images and saved for later use.

---

## 🛠 Technologies Used

* **Python 3**
* **TensorFlow / Keras**
* **OpenCV** (for image preprocessing)
* **PIL (Pillow)**
* **NumPy**
* **scikit-learn**

---

## 📂 Project Structure

```
├── datasets/
│   ├── no/         # Images without tumor
│   └── yes/        # Images with tumor
├── BrainTumor10EpochsCategorical.h5   # Saved trained model
├── brain_tumor_cnn.py                 # Main training script
└── README.md
```

---

## ⚙️ How It Works

1. **Image Preprocessing**

   * Reads images using OpenCV.
   * Converts them to RGB and resizes to **64×64** pixels.
   * Normalizes pixel values between 0–1.
   * Labels: `0 = No Tumor`, `1 = Tumor`.

2. **Dataset Split**

   * Train-Test split (80:20).
   * Labels converted to categorical format.

3. **CNN Model**

   * **Conv2D + MaxPooling** layers for feature extraction.
   * **Dense + Dropout** layers to reduce overfitting.
   * **Softmax activation** for binary classification.

4. **Training**

   * Optimizer: **Adam**
   * Loss: **Categorical Crossentropy**
   * Epochs: **10**
   * Batch size: **16**

5. **Output**

   * Trained model saved as `BrainTumor10EpochsCategorical.h5`.

---

## 📈 Model Architecture

* **Input:** (64, 64, 3) images
* **Conv2D → ReLU → MaxPooling** (3 blocks)
* **Flatten → Dense (64) → ReLU → Dropout (0.5)**
* **Dense (2) → Softmax**

---

## 🚀 How to Run

1. Clone this repo:

2. Install dependencies:

   ```bash
   pip install tensorflow keras opencv-python pillow numpy scikit-learn
   ```

3. Place dataset in `datasets/no/` and `datasets/yes/`.

4. Run training:

   ```bash
   python brain_tumor_cnn.py
   ```

5. The trained model will be saved as:

   ```
   BrainTumor10EpochsCategorical.h5
   ```

---

## 📌 Future Improvements

* Add **data augmentation** for better generalization.
* Increase epochs & tune hyperparameters.
* Deploy model as a **Flask/Django API**.
* Build a simple **web app** for MRI upload & prediction.

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repo and create a pull request.

---
