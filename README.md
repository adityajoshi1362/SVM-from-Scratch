# Primal Form SVM Implementation

## Project Overview

This project is an implementation of the Primal Form of Support Vector Machine (SVM) from scratch using Python. It focuses on understanding the mathematical foundations and building the model without relying on high-level libraries like scikit-learn for the core algorithm.

## What is the Primal Form of SVM?

The Primal Form of SVM is a direct formulation of the SVM optimization problem. It aims to minimize the following objective:

\[ \frac{1}{2} ||w||^2 + C \sum_{i=1}^{n} \max(0, 1 - y_i (w \cdot x_i + b)) \]

Where:
- \( w \) = Weight vector
- \( b \) = Bias term
- \( C \) = Regularization parameter
- \( x_i, y_i \) = Data points and labels

The primal form directly optimizes the weights using gradient descent.

## Dataset Information

The implementation has been tested on various datasets such as:
- **Linearly separable data**
- **make_moons dataset** (non-linear data with crescent shapes)
- **make_circles dataset** (non-linear circular data)

Ensure your data is normalized for optimal performance.

## Results/Performance

- Achieves high accuracy on well-separated datasets.
- Successfully handles challenging non-linear datasets like `make_moons` and `make_circles` with appropriate tuning.

## Future Improvements

- Implement RBF Kernel for better handling of non-linear datasets.
- Add cross-validation for improved model generalization.
- Include visualization tools to illustrate decision boundaries.

