import numpy as np


def extend_to_basis(orthonormal_set):
    """Extends the orthonormal set to a basis of the space."""
    dim = orthonormal_set.shape[1]  # Number of rows gives the dimension

    # Initialize the full basis list with the given orthonormal vectors
    full_basis = list(orthonormal_set)

    def normalize(v, basis):
        for b in basis:
            v -= np.dot(v, b) * b
        norm_v = np.linalg.norm(v)
        return v / norm_v if norm_v != 0 else None

    # Continue adding vectors until the basis is complete
    while len(full_basis) < dim:
        v = np.random.rand(dim)  # Generate a random vector
        v = normalize(v, full_basis)
        if v is not None:
            full_basis.append(v)

    # Return the full orthonormal basis as a matrix
    return np.array(full_basis).T