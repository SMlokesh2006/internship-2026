# Detailed Documentation: E-Waste Classification Project

## 1. Project Overview
This project focuses on the automated classification of Electronic Waste (E-Waste) into 10 distinct categories using deep learning. Developed as part of the Edunet Internship Program, the primary objective is to assist in effective recycling and waste management.

The model is built using **TensorFlow/Keras** and employs Transfer Learning with the **EfficientNetV2B0** architecture. The final trained model is deployed through a web-based interface built with **Gradio**, allowing users to upload images and instantly get classification results.

## 2. Environment Setup & Dependencies
The project relies on standard data science and deep learning libraries.

**Core Libraries:**
- `tensorflow`: Core deep learning framework.
- `numpy`: Numerical operations and matrix handling.
- `matplotlib` & `seaborn`: Plotting loss/accuracy curves and confusion matrices.
- `scikit-learn`: Evaluation metrics (classification report, confusion matrix).
- `gradio`: Creating the web interface for the model.
- `pillow (PIL)`: Image processing.

## 3. Dataset Information
The dataset consists of images categorized into 10 classes of E-Waste:
1. Battery
2. Keyboard
3. Microwave
4. Mobile
5. Mouse
6. PCB (Printed Circuit Board)
7. Player
8. Printer
9. Television
10. Washing Machine

Images are loaded from local directories (`train`, `val`, `test`) using `tf.keras.utils.image_dataset_from_directory` and are resized to **128x128 pixels** with a **batch size of 32**.

## 4. Data Preprocessing & Augmentation
To prevent overfitting and improve the generalization of the model, a Sequential Data Augmentation pipeline is integrated into the model architecture:
- **RandomFlip("horizontal")**: Flips images randomly.
- **RandomRotation(0.1)**: Rotates images by up to 10%.
- **RandomZoom(0.1)**: Zooms into images by up to 10%.
- **RandomContrast(0.1)**: Adjusts image contrast randomly.
- **RandomBrightness(0.2)**: Adjusts image brightness randomly.

## 5. Model Architecture
The network is based on **EfficientNetV2B0** (pre-trained on ImageNet).

**Architecture Breakdown:**
1. **Input Layer**: Shape `(128, 128, 3)`.
2. **Data Augmentation Layer**: Applies transformations dynamically during training.
3. **Base Model (EfficientNetV2B0)**: 
   - The first 100 layers are **frozen** to retain pre-trained feature extraction capabilities and speed up training.
   - Layers after the 100th are kept **trainable** for fine-tuning.
4. **GlobalAveragePooling2D**: Reduces spatial dimensions to a 1D vector (size 1280).
5. **Dropout (0.2)**: Randomly drops 20% of neurons to prevent overfitting.
6. **Dense Output Layer**: 10 neurons with `softmax` activation for predicting class probabilities.

**Parameters:**
- Total parameters: ~16.27 Million
- Trainable parameters: ~5.17 Million
- Non-trainable parameters (frozen): ~763K

## 6. Training Methodology
- **Optimizer**: Adam with an initial learning rate of `0.0001`.
- **Loss Function**: `SparseCategoricalCrossentropy` (since labels are integer-encoded).
- **Metrics**: `Accuracy`.
- **Epochs**: Trained for up to 10 epochs.

**Callbacks used:**
- **EarlyStopping**: Monitors `val_loss`, stops training if it doesn't improve for 3 epochs, and restores the best weights.
- **ReduceLROnPlateau**: Halves the learning rate (`factor=0.5`) if `val_loss` plateaus for 3 epochs.

## 7. Evaluation & Visualization
- The notebook plots **Training vs Validation Accuracy** and **Training vs Validation Loss** using Matplotlib.
- The model's performance on the test set is evaluated numerically.
- A **Classification Report** (Precision, Recall, F1-Score) and a **Confusion Matrix** (visualized via Seaborn heatmap) are generated to thoroughly analyze the model's predictive power across individual classes.

## 8. Web Interface Deployment (Gradio)
The project provides an interactive web application to demonstrate the model's capabilities:
- **Framework**: Gradio.
- **Functionality**: Users upload an image, which is resized to 128x128, converted to a NumPy array, preprocessed (using EfficientNet's `preprocess_input`), and passed to the model.
- **Output**: The application displays the predicted class name along with the confidence percentage (e.g., `Confidence: 98.45%`).
- **Mode**: The Gradio interface runs in `live=True` mode, meaning the prediction updates automatically when a new image is loaded.

## 9. Included Files Summary
- `E-Waste classification Model.ipynb`: The complete executable Jupyter Notebook containing data pipeline, model definition, training, evaluation, and deployment code.
- `base_model_summary.txt`: Structural summary of the EfficientNetV2B0 base model.
- `model_summary.txt`: Structural summary of the final fine-tuned Sequential model.
- `README.md`: Short, introductory overview of the repository.
- `PPT E waste classification by Kavin Raj N V.pptx`: Presentation slides.
- `Sample screen shot.pdf`: PDF containing screenshots of model operation/Gradio interface.
*(Note: The trained `.keras` model file is excluded due to size limitations, as mentioned in the project's README).*
