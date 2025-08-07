# SCT_ML_03

Cat vs Dog Image Classifier using SVM + MobileNetV2

This project implements an efficient **image classification system** that distinguishes between **cats and dogs** using a **Support Vector Machine (SVM)** classifier trained on features extracted from **MobileNetV2** (a lightweight CNN). It combines the power of deep learning for feature extraction and the simplicity of SVM for fast, accurate classification.

---

##   Overview

- Extracts deep image features using pretrained **MobileNetV2** (transfer learning).
- Trains a **Linear SVM** on these features.
- Achieves **98% accuracy** on a dataset of 25,000 labeled images.
- Fast inference, suitable for deployment or real-time apps.

---

##  Technologies Used

- Python 3.10+
- TensorFlow / Keras
- Scikit-learn
- NumPy, Matplotlib, Seaborn
- OpenCV (for preprocessing)
- Joblib / Pickle (for saving models)

---

##  Project Structure

Image-Classifier/
├── svm_mobilenet_model.pkl # Trained SVM model (joblib format)
├── svm_final_model.pkl # Trained SVM model (pickle format, for better compatibility)
├── training_images/ # Folder containing training images (cats and dogs)
├── predict.py # Python script to predict a single image
├── SVM_train.py # Training script
└── README.md # Project description


---

##  Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/cat-dog-svm-classifier.git
   cd cat-dog-svm-classifier
2.Install required dependencies:
pip install tensorflow scikit-learn numpy matplotlib joblib
3. Make sure your training images are placed in:
training_images/train/
    cat.1.jpg
    dog.1.jpg
    ...
 Training
To train the SVM model:

python SVM_train.py
This will:

Load images

Preprocess and extract features via MobileNetV2

Train an SVM classifier

Save the model as .pkl

