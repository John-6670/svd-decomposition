# svd-decomposition

An implementation of Singular Value Decomposition (SVD) in Python, along with some of its applications.

## Overview

This repository demonstrates the computation of Singular Value Decomposition (SVD) in Python, leveraging `numpy.linalg.eigh`
for eigenvalue decomposition of symmetric matrices and `numpy.linalg.norm` for vector norms. The primary goal is to provide a
clear and educational implementation of SVD using established numerical routines for core linear algebra operations, rather than
a fully from-scratch implementation of all underlying algorithms. This approach balances clarity and efficiency, making the code
easier to understand while still providing reasonable performance.

## Implementation

The SVD is computed using the following steps:

1.  Compute the eigenvalues and eigenvectors of A<sup>T</sup>A using `numpy.linalg.eigh`.
2.  Construct Σ (the diagonal matrix of singular values) and V<sup>T</sup> (the transpose of the right singular vectors) from the eigenvalues and eigenvectors.
3.  Compute U (the left singular vectors) using the relationship U = (A * V) / Σ.

The implementation is encapsulated within an `SVD` class for better organization and maintainability.

Key features:

*   **Uses `numpy.linalg.eigh` and `numpy.linalg.norm`:** Leverages NumPy's efficient routines for core linear algebra operations.
*   **`SVD` Class:** Encapsulates the SVD computation logic.
*   **Handles Rank Deficiency:** Includes a function to extend the set of left singular vectors to a full orthonormal basis when the matrix is rank-deficient.
*   **Clear Documentation:** The code is thoroughly documented with docstrings to explain the purpose of each function and method.

## Applications

This repository also includes examples demonstrating the use of SVD in various applications:

*   **Image Compression:** Demonstrates how SVD can be used to compress images by keeping only the most significant singular values.
*   **Basic Usage:** Shows how to compute the SVD of a simple matrix and reconstruct the original matrix.

More applications will be added in the future.

## Getting Started

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/John-6670/svd-decomposition
    ```

2.  **Navigate to the project directory:**

    ```bash
    cd svd-decomposition
    ```

3.  **(Optional) Create a virtual environment (recommended):**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Linux/macOS
    .venv\Scripts\activate     # On Windows
    ```

4.  **Install requirements:**

    ```bash
    pip install -r requirements.txt
    ```