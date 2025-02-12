<!-- Intro-->

<!--
*  This README.md is inspired by the Project-README-Template!
*  (https://github.com/YousefIbrahimismail/Project-README-Template)
*  Thanks YousefIbrahimismail!
-->

<!-- Project title -->
<div align="center">
<img src="https://readme-typing-svg.demolab.com?font=Merriweather&20Bold&size=28&duration=3000&pause=7000&vCenter=true&multiline=true&width=900&lines=Applying+The+Analog+Activation+Function+to+Image+Classification">
</div>
<!-- Shields Section-->
<div align="center">
    <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/Hippocratic_License_V3-blue.svg"></a>
    <a href="https://paperswithcode.com/paper/evaluating-the-performance-of-taaf-for-image"><img alt="Papers With Code" src="https://img.shields.io/badge/Papers_with_code-grey?logo=paperswithcode&logoColor=21CBCE"></a>
    <a href="https://www.academia.edu/127610553/The_Analog_Activation_Function_TAAF_of_Emergent_Linear_Systems_Evaluating_the_Performance_of_The_Analog_Activation_Function_TAAF_on_MNIST_and_CIFAR_10_Datasets"><img alt="Papers With Code" src="https://img.shields.io/badge/Academia.edu-%23003865?logo=Academia"></a>
</div>


## About
<!-- information about the project -->

This repository contains code and resources for the paper: **"The Analog Activation Function (TAAF) of Emergent Linear Systems: Evaluating the Performance of The Analog Activation Function (TAAF) on MNIST and CIFAR-10 Datasets."**

This paper introduces the **Analog Activation Function (TAAF)**, a novel activation function designed for neural networks, inspired by the principles of emergent linearity. We evaluate TAAF's performance on image classification tasks using the MNIST and CIFAR-10 datasets, comparing it against standard activation functions.

**Key Findings:**

*   **MNIST Dataset:** TAAF achieves a test accuracy of **99.39%** on MNIST.
*   **CIFAR-10 Dataset:** TAAF achieves a test accuracy of **79.37%** on CIFAR-10.

This repository provides the code to reproduce our experiments and explore the TAAF activation function.

## Key Features

This repository includes:

*   **`src/` Directory:**
    *   **`mnist/`:** Code for training and evaluating models on MNIST.
    *   **`cifar10/`:** Code for training and evaluating models on CIFAR-10.
    *   **`activation_functions.py`:** Python file defining the `TAAF` and `ELU` activation function classes.
    *   **`utils.py`:** Utility functions and potentially common model components.
*   **`tests/` Directory:**
    *   **`mnist/`:** Saved model checkpoints for MNIST experiments.
    *   **`cifar10/`:** Saved model checkpoints for CIFAR-10 experiments.
*   **`LICENSE`:** License file for the repository.
*   **`README.md`:** This README file.
*   **`requirements.txt`:** List of Python package dependencies.

## Getting Started

### Prerequisites

*   **Python 3.x**
*   **PyTorch** (>= version mentioned in your code, if any)
*   **Torchvision**
*   **tqdm**
*   **Other packages** as listed in `requirements.txt`.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [repository-url]
    cd [repository-directory]
    ```
    Replace `[repository-url]` and `[repository-directory]`.

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Code

**MNIST Experiments:**

1.  Navigate to the MNIST directory:
    ```bash
    cd src/mnist
    ```

2.  Run the training script:
    ```bash
    python train.py
    ```

**CIFAR-10 Experiments:**

1.  Navigate to the CIFAR-10 directory:
    ```bash
    cd src/cifar10
    ```

2.  Run the training script:
    ```bash
    python train.py
    ```

**Model Checkpoints:**

Trained model checkpoints are saved in the `tests/mnist/` and `tests/cifar10/` directories.

## Datasets

*   **MNIST Dataset:** Automatically downloaded by torchvision.
*   **CIFAR-10 Dataset:** Automatically downloaded by torchvision.

## Model and Training Details

The CNN architecture and training parameters are detailed in the paper.

**Key Training Parameters:**

*   **Optimizer:** Adam
*   **Learning Rate:** 0.001
*   **Loss Function:** Cross-Entropy Loss
*   **Batch Size:** 64
*   **Epochs:** 10 for MNIST, 20 for CIFAR-10

## Results

Detailed results are in the paper.

**Summary of Key Test Accuracies Achieved with TAAF:**

*   **MNIST:** ~99.39%
*   **CIFAR-10:** ~79.37%

## File Structure
repository-directory/
├── src/
│ ├── mnist/
│ │ └── train.py
│ ├── cifar10/
│ │ └── train.py
├── tests/
│ ├── mnist/
│ │ └── mnist_TAAF_model.pth
│ └── cifar10/
│   └── cifar10_TAAF_model.pth
├── LICENSE
├── README.md
└── requirements.txt

## Citation
Bryn T. Chatfield. (2025). The Analog Activation Function (TAAF) of Emergent Linear Systems: Evaluating the Performance of The Analog Activation Function (TAAF) on MNIST and CIFAR-10 Datasets. OpenASCI & Genova Laboratories Research & Development Div.

## License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file.

## Contact
Bryn T. Chatfield
[LinkedIn Profile](https://www.linkedin.com/in/bryn-chatfield/)
OpenASCI & GenoLabs

---

<!-- Table of Contents-->
<dev display="inline-table" vertical-align="middle">
<table align="center" vertical-align="middle">
    <tr>
        <td><a href="#about">About</a></td>
        <td><a href="#key-features">Key Features</a></td>
        <td><a href="#getting-started">Getting Started</a></td>
        <td><a href="#datasets">Datasets</a></td>
        <td><a href="#model-and-training-details">Model Details</a></td>
        <td><a href="#results">Results</a></td>
        <td><a href="#file-structure">File Structure</a></td>
        <td><a href="#license">License</a></td>
        <td><a href="#citation">Citation</a></td>
        <td><a href="#contact">Contact</a></td>
    </tr>
</table>
</dev>

<p align="right"><a href="#top">back to top ⬆️</a></p>

## Acknowledgments

*   Inspired by the Project-README-Template by YousefIbrahimismail.
*   [Make a Readme](https://www.makeareadme.com/)
*   [Shields](https://shields.io/)
*   [SVG README](https://readme-typing-svg.demolab.com/demo/)

<p align="right"><a href="#top">back to top ⬆️</a></p>

## Feedback

> Contributions are encouraged! Please feel free to open a [Pull Request](https://github.com/bryn-gnolbs/TAAF-for-Image-Classification/pulls).

<p align="right"><a href="#top">back to top ⬆️</a></p>