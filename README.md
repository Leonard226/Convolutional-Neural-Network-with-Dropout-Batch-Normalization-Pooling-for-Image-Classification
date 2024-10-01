# Convolutional Neural Network w/ Dropout, Batch Normalization, Pooling for Image Classification
## MNIST Image Classification using a Convolutional Neural Network (CNN)

#### Overview

This project implements a **Convolutional Neural Network (CNN)** for image classification on the MNIST dataset. The network classifies handwritten digits (0–9) with an accuracy of 99% on the training data, leveraging advanced deep learning techniques in **PyTorch**.

#### Key Components

- **Dataset**: MNIST dataset (grayscale, 28x28 pixels), loaded with `DataLoader` for efficient batch processing.
- **Model**:
  - **Multiple Convolutional Layers**: Feature extraction from input images using several convolutional layers.
  - **ReLU Activations**: Non-linearity introduced between layers for better learning.
  - **Max Pooling**: Downsampling after convolutional layers to reduce spatial dimensions.
  - **Dropout**: Applied to prevent overfitting during training.
  - **Multiple Linear Layers**: Dense layers for classification after feature extraction.
  - **Batch Normalization**: Used for faster convergence and stable training.
- **Training**:
  - **Optimizer**: Stochastic Gradient Descent (SGD) with learning rate 1e-3.
  - **Loss Function**: Cross-Entropy Loss for multiclass classification.
  - **GPU** Training: Utilizes CUDA or MPS for faster computations.
  - **Training Loop**: 50 epochs of forward passes, backpropagation, and weight updates.
- **Evaluation**: Model performance is evaluated based on accuracy and average loss on the test set.

#### External Image Classification

The project supports classifying external images by preprocessing them (resizing, grayscaling) and predicting their digit class.

#### Visualizations

- **Sample Predictions**: Visualizes training samples and model predictions using `matplotlib`.
