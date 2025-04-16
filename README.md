# KernelNet

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [License](#license)

## Overview
**KernelNet** is a neural network API implemented from scratch written in C++ and CUDA, offering streamlined components for building, training, and evaluating deep learning models. The framework includes its own tensor operations with automatic differentiation, suitable for developing custom neural architectures. 

## Features
### Tensor Operations & Autograd:
- Custom tensor operations for numerical arithmetic computations include element-wise addition, subtraction, and multiplication; matrix multiplication and transpose; as well as scalar multiplication, broadcast addition, summation, and argmax.
- Built-in automatic differentiation as the backbone of gradient backward propagation.
### Neural Network Building Blocks:
- Dense Layers: Fully connected layers for standard feedforward networks.
- Convolutional Layers (Conv2D): Convolution operations for feature extraction in images.
- MaxPooling Layers (MaxPool2D): Downsampling operations to reduce spatial dimensions following convolutions.
- Recurrent Layers (LSTM): LSTM cells with a sequential wrapper to process time-series or sequential data.
- Embedding Layers: Components that transform categorical data into dense vector representations.
### Activation Functions:
- Tanh: Hyperbolic tangent activation for non-linear transformations.
- Sigmoid: Function that maps outputs to a [0, 1] range.
- ReLU: Rectified Linear Unit for gradient propagation in deep networks.
- Softmax: Normalizes outputs for multi-class classification tasks.
### Optimization & Loss Functions:
- SGD Optimizer: Implements stochastic gradient descent with optional gradient clipping.
- Mean Squared Error and Cross-Entropy.
### Hardware Support:
-  Supports both CPU-based computation and GPU-optimized parallel implementations using CUDA-enabled hardware.
### Benchmarking:
-  Provides integrated upstream data-loading and preprocessing pipelines that allows benchmarking both the time efficiency and accuracy of convolutional and recurrent models built using the **KernelNet** architecture on **CIFAR10** and **PennTree** Dataset.

## Installation
- Obtain the **KernelNet** [`packages`](https://github.com/ron-che-debugger/KernelNet/releases/tag/installation), which contain the pre-built libraries [`kernelnet.dll`](bin/kernelnet.dll), [`kernelnet.lib`](bin/kernelnet.lib), and an [`/include`](bin/include) folder.  
- Place [`kernelnet.dll`](bin/kernelnet.dll) in your executable's folder or system path, and configure your linker to reference [`kernelnet.lib`](bin/kernelnet.lib).
- Add the [`/include`](bin/include) folder to your project's include directories. In your source files, simply use:
    ```cpp
    #include "kernelnet.hpp"
    ```
    to access the functionalities. For detailed function signatures and class descriptions, check the [`KernelNet API`](https://ronghanche.com/kernelnet-main).

## License
This project is licensed under the MIT License. See the [`LICENSE`](LICENSE) file for details.