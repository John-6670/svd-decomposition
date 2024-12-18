import numpy as np
from utils import extend_to_basis

class SVD:
    """
    Implements Singular Value Decomposition (SVD) of a matrix A.

    This class computes the SVD of a real matrix A using the eigenvalue decomposition
    of A^T * A. It returns the three matrices U, Sigma, and VT such that:

    A = U * Sigma * VT

    Attributes:
        __A (np.ndarray): The input matrix A.
        __tol (float): Tolerance for considering singular values as zero.
        __U (np.ndarray): The left singular vectors of A (orthonormal columns).
        __singulars (np.ndarray): The singular values of A (on the diagonal of Sigma).
        __VT (np.ndarray): The right singular vectors of A^T (orthonormal rows).
    """

    def __init__(self, A, tol=1e-4):
        """
        Initializes the SVD class.

        Args:
            A (np.ndarray): The input matrix A (2D array).
            tol (float, optional): Tolerance for considering singular values as zero.
                Defaults to 1e-4.
        """
        self.__A = np.array(A, dtype=float)  # Ensure float type for calculations
        self.__tol = tol
        self.__U = None
        self.__singulars = None
        self.__VT = None

    def __construct_sigma_vt(self):
        """
        Constructs Sigma and V^T from eigenvalues and eigenvectors of A^T * A.

        Calculates the eigenvalues and eigenvectors of A^T * A and extracts the
        singular values (square root of non-zero eigenvalues) and right singular vectors
        (transposed eigenvectors).
        """
        eigenvalues, eigenvectors = np.linalg.eigh(np.dot(self.__A.T, self.__A))
        # Sort eigenvalues and eigenvectors in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Filter out eigenvalues smaller than the tolerance
        self.__singulars = np.sqrt(eigenvalues[eigenvalues > self.__tol])
        self.__VT = eigenvectors.T

    def __calculate_u(self):
        """
        Calculates the left singular vectors U using U = (A * V) / Sigma.

        Computes the left singular vectors (U) by multiplying the input matrix A
        with the right singular vectors (transposed VT) and dividing by the corresponding
        singular values (on the diagonal of Sigma).
        """
        u_set = np.array([np.dot(self.__A, self.__VT.T[:, i]) / self.__singulars[i] for i in range(len(self.__singulars))]).T

        # Check if U has fewer vectors than the number of rows in A (potential rank deficiency)
        if len(u_set.T) < self.__A.shape[0]:
            # Extend the set with additional orthogonal vectors to form a basis
            u_set = extend_to_basis(u_set.T)

        self.__U = u_set

    def compute_svd(self):
        """
        Computes the SVD of the input matrix A.

        Calculates the SVD of A and returns the three matrices U, Sigma, and VT.

        Returns:
            tuple: (U, Sigma, VT)
                - U (np.ndarray): The left singular vectors of A.
                - Sigma (np.ndarray): The singular values of A (diagonal matrix).
                - VT (np.ndarray): The right singular vectors of A^T.
        """

        self.__construct_sigma_vt()
        self.__calculate_u()

        # Create a diagonal matrix to store singular values
        sigma = np.zeros(self.__A.shape)
        np.fill_diagonal(sigma, self.__singulars)

        return self.__U, sigma, self.__VT
