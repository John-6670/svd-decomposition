# svd-decomposition

This project implements the Singular Value Decomposition (SVD) algorithm in Python. The SVD is a fundamental matrix factorization technique used in various applications such as signal processing, statistics, and machine learning.

## Installation

To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

## Usage
To use the SVD, you can create an instance of the SVD class and call the compute_svd method. Here is an example
```bash
from svd.svd import SVD

# Example matrix
A = [[1, 2], [3, 4], [5, 6]]

# Create an SVD instance
svd = SVD(A)

# Compute the SVD
U, Sigma, VT = svd.compute_svd()

print("U:", U)
print("Sigma:", Sigma)
print("VT:", VT)
```

## Applications
Applications of SVD include, but are not limited to (to be added):

- Data compression (image compression, sound compression, etc.)
- Noise Reduction and Signal Processing