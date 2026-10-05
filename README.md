# Handwritten Digit Recognition

## Project Overview
This project recognizes handwritten digits using the MNIST dataset and a neural network built with TensorFlow and Keras.

## Objective
The main objective is to identify handwritten digits from image data accurately.

## Dataset
The project uses the MNIST handwritten digit dataset.

Dataset: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv

## Technologies Used
- Python
- NumPy
- TensorFlow
- Keras
- Matplotlib

## Methodology
1. Load the MNIST dataset.
2. Preprocess and normalize the image data.
3. Build a neural network model.
4. Train the model using training data.
5. Evaluate the model using test data.
6. Visualize predictions.

## Model Architecture
- Flatten Layer
- Dense Layer: 128 neurons, ReLU
- Dense Layer: 64 neurons, ReLU
- Output Layer: 10 neurons, Softmax

## Result
The model achieved approximately **97.04% test accuracy**.

## How to Run

Install the required libraries:

```bash
pip install numpy matplotlib tensorflow
