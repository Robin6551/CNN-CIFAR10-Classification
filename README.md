CNN-CIFAR10-Classification
📘 Computer Vision and AI (CYS22202) - Project Report Code
This repository contains the Jupyter Notebook code for the assignment report on implementing a Convolutional Neural Network (CNN) for multi-class object classification using the CIFAR-10 dataset.

1. Project Objective
The primary goal of this project was to design, implement, and evaluate a custom CNN architecture capable of classifying low-resolution (32x32 pixel) images into one of 10 distinct categories.

Dataset Overview
Dataset: CIFAR-10

Total Images: 60,000 (50,000 training, 10,000 testing)

Classes: Airplane, Car, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck

2. Model Architecture
The implemented model follows a standard deep learning approach for image feature extraction and classification.

Layer Type

Configuration

Output Shape (Approx.)

Input

32×32×3 (RGB)

N/A

Conv2D + ReLU

32 filters, (3,3) kernel

30×30×32

MaxPooling

(2,2) pool size

15×15×32

Conv2D + ReLU

64 filters, (3,3) kernel

13×13×64

MaxPooling

(2,2) pool size

6×6×64

Conv2D + ReLU

64 filters, (3,3) kernel

4×4×64

Flatten

-

1024

Dense + ReLU

64 neurons

64

Dense + Softmax

10 neurons (Output)

10

3. Setup and Execution
To run the project notebook (cnn_cifar10_classifier.ipynb), you need Python and the necessary libraries installed.

Prerequisites
Python 3.x

Jupyter Notebook or JupyterLab (standard in an Anaconda installation)

Installation
It is highly recommended to use a virtual environment or an Anaconda environment (like (base) or a dedicated environment) for installation.

Install Required Libraries:

pip install tensorflow numpy matplotlib scikit-learn

Run the Notebook:

Start Jupyter Notebook from your terminal:

jupyter notebook

Open the cnn_cifar10_classifier.ipynb file in your browser and run all cells sequentially (Cell -> Run All).

4. Key Results Summary
After 10 training epochs, the model achieved the following performance metrics on the unseen test set:

Metric

Result (Approx.)

Test Accuracy

~72%

Precision (avg)

~70%

Recall (avg)

~71%

F1-Score (avg)

~70%

The notebook includes plots detailing the training and validation accuracy/loss history, as well as the full Confusion Matrix showing classification performance per class.

5. Challenges and Future Work
The primary challenge encountered was overfitting, indicated by the divergence between training and validation accuracy after several epochs.

Future work would focus on:

Implementing Data Augmentation (e.g., random flips and rotations).

Exploring Transfer Learning using pre-trained backbones (like VGG16) to leverage features learned from larger datasets.
