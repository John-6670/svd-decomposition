import numpy as np
from numpy.linalg import eigh
from .utils import extend_to_basis


class SVD:
    def __init__(self, A, tol=1e-4):
        self.__A = np.array(A, dtype=float)  # Ensure float type for calculations
        self.__tol = tol
        self.__U = None
        self.__singulars = None
        self.__VT = None

    def __construct_sigma_vt(self):
        """Constructs Sigma and V^T from eigenvalues and eigenvectors."""
        eigenvalues, eigenvectors = eigh(np.dot(self.__A.T, self.__A))
        idx = np.argsort(eigenvalues)[::-1]
        # Reorder the eigenvalues and eigenvectors accordingly
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        eigenvalues = eigenvalues[eigenvalues > self.__tol]

        self.__singulars = np.sqrt(eigenvalues)
        self.__VT = eigenvectors.T

    def __calculate_u(self):
        """Calculates U using U = (A * V) / Sigma."""
        u_set = np.array([np.dot(self.__A, self.__VT.T[:, i]) / self.__singulars[i] for i in range(len(self.__singulars))]).T

        # Check if u_set has fewer vectors than expected
        if len(u_set.T) < self.__A.shape[0]:
            # if the vectors are not sufficient, extend the set
            u_set = extend_to_basis(u_set.T)

        self.__U = u_set

    def compute_svd(self):
        """Computes the SVD of A."""
        # max_A = np.max(np.abs(self.__A))
        # self.__A = self.__A / max_A if max_A != 0 else self.__A # Normalize A
        self.__construct_sigma_vt()
        self.__calculate_u()
        sigma = np.zeros(self.__A.shape)
        np.fill_diagonal(sigma, self.__singulars)
        return self.__U, sigma, self.__VT
