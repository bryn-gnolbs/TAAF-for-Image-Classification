# The Analog Activation Function (TAAF) of Emergent Linear Systems

[![Paper](link-to-your-paper-here)](link-to-your-paper-here) <!-- Replace with a link to your paper if available (e.g., ArXiv, website) -->
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) <!-- Replace with your license badge if applicable -->

## Overview

This repository contains code and resources for the paper: **"The Analog Activation Function (TAAF) of Emergent Linear Systems: Evaluating the Performance of The Analog Activation Function (TAAF) on MNIST and CIFAR-10 Datasets."**

This paper introduces the **Analog Activation Function (TAAF)**, a novel activation function designed for neural networks, inspired by the principles of emergent linearity.  We evaluate TAAF's performance on image classification tasks using the MNIST and CIFAR-10 datasets, comparing it against standard activation functions.

**Key Findings:**

*   **MNIST Dataset:** TAAF achieves a test accuracy of **99.39%** on MNIST, slightly outperforming standard activation functions like ReLU, GELU, Sigmoid, and Tanh in our experiments using a simple Convolutional Neural Network (CNN) architecture.
*   **CIFAR-10 Dataset:** TAAF demonstrates a significant improvement in generalization capability on the more complex CIFAR-10 dataset, achieving a test accuracy of **79.37%**, outperforming ELU and other common activation functions (GELU, ReLU, Sigmoid, Tanh) in the same CNN architecture by a notable margin.

This repository provides the code to reproduce our experiments and explore the TAAF activation function.

## Key Features

This repository includes:

*   **`src/` Directory:**
    *   **`mnist/`:** Code for training and evaluating models with TAAF and other activation functions on the MNIST dataset.
    *   **`cifar10/`:** Code for training and evaluating models with TAAF and ELU on the CIFAR-10 dataset.
    *   **`activation_functions.py`:** Python file defining the `TAAF` and `ELU` activation function classes in PyTorch.
    *   **`utils.py`:** Utility functions and potentially common model components.
*   **`tests/` Directory:**  (Potentially - if you have separate test scripts)
    *   **`mnist/`:** Saved model checkpoints for MNIST experiments.
    *   **`cifar10/`:** Saved model checkpoints for CIFAR-10 experiments.
*   **`LICENSE`:**  License file for the repository (e.g., MIT License).
*   **`README.md`:** This README file.
*   **(Potentially) `requirements.txt`:**  List of Python package dependencies.

## Getting Started

### Prerequisites

*   **Python 3.x**
*   **PyTorch** (>= version mentioned in your code, if any)
*   **Torchvision**
*   **tqdm** (for progress bars)
*   **(Potentially) Other packages** as listed in `requirements.txt` if you include one.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [repository-url]
    cd [repository-directory]
    ```
    Replace `[repository-url]` with the actual URL of your GitHub repository and `[repository-directory]` with the name of the cloned directory.

2.  **Install Dependencies (Optional - if you have `requirements.txt`):**
    ```bash
    pip install -r requirements.txt
    ```
    Alternatively, install dependencies manually if you don't have a `requirements.txt` file (e.g., `pip install torch torchvision tqdm`).

### Running the Code

**MNIST Experiments:**

1.  Navigate to the MNIST directory:
    ```bash
    cd src/mnist
    ```

2.  Run the training script for TAAF (or other activations):
    ```bash
    python train.py
    ```
    You can modify the `train.py` script to experiment with different activation functions (TAAF, ELU, etc.) by changing the `activation_type` variable in the script.

**CIFAR-10 Experiments:**

1.  Navigate to the CIFAR-10 directory:
    ```bash
    cd src/cifar10
    ```

2.  Run the training script for TAAF (or ELU):
    ```bash
    python train.py
    ```
    Similarly, modify `train.py` in this directory to choose between `TAAF` and `ELU` activation functions.

**Model Checkpoints:**

Trained model checkpoints are saved in the `tests/mnist/` and `tests/cifar10/` directories after training.

## Datasets

*   **MNIST Dataset:**  The MNIST dataset of handwritten digits is automatically downloaded by torchvision when you run the training scripts if it's not already present in the `./data` directory.
*   **CIFAR-10 Dataset:** The CIFAR-10 dataset is also automatically downloaded by torchvision when you run the CIFAR-10 training script if it's not already present in the `./data` directory.

## Model and Training Details

The CNN architecture and training parameters used in these experiments are detailed in the paper: **"The Analog Activation Function (TAAF) of Emergent Linear Systems: Evaluating the Performance of The Analog Activation Function (TAAF) on MNIST and CIFAR-10 Datasets."**

**Key Training Parameters:**

*   **Optimizer:** Adam
*   **Learning Rate:** 0.001
*   **Loss Function:** Cross-Entropy Loss
*   **Batch Size:** 64
*   **Epochs:** 10 for MNIST, 20 for CIFAR-10

Please refer to the paper for a complete description of the model architecture, training process, and hyperparameter choices.

## Results

Detailed results, including training and testing performance, comparisons with other activation functions, and visualizations, are presented in the paper.

**Summary of Key Test Accuracies Achieved with TAAF:**

*   **MNIST:** ~99.39%
*   **CIFAR-10:** ~79.37%

See the paper for in-depth analysis and comparisons.

## File Structure