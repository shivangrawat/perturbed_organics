import torch
import numpy as np
import math
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.signal import argrelextrema


def dynm_fun(f):
    """A wrapper for the dynamical function"""

    def wrapper(self, t, x):
        new_fun = lambda t, x: f(self, t, x)
        return new_fun(t, x)

    return wrapper


def make_spike_train(simulation, dt=1e-4):
    """
    This function receives an inhomogeneous firing rate as a function of time and dt.
    We approximae the Poisson process by a Bernoulli process.
    :return: The spike train in binary.
    """
    uniform_rand = torch.rand_like(simulation)
    spike_prob = simulation * dt
    spike_train = uniform_rand < spike_prob
    return spike_train


def generate_stable_system(n):
    """
    This function returns a randomly generated stable linear dynamical system (real part of the eigenvalues are negative)
    :param n: dimension of the system
    :return:
    """
    A = torch.randn(n, n)
    L, V = torch.linalg.eig(A)
    L = -torch.abs(torch.real(L)) + 1j * torch.imag(L)
    return torch.real(V @ torch.diag(L) @ torch.linalg.inv(V))


def cholesky_decomposition(A):
    """
    This function returns the cholesky decomposition of a matrix A.
    :param A: The matrix to be decomposed.
    :return: The cholesky decomposition of A.
    """
    L = torch.linalg.cholesky(A)
    return L


### Functions to sample different types of matrices


def generate_matrix(N, matrix_type, **kwargs):
    """
    Generate different types of matrices based on the matrix_type.
    
    :param N: Size of the matrix.
    :param matrix_type: Type of matrix to generate. Options are 'goe', 'goe_symmetric', or 'power_law'.
    :param kwargs: Additional parameters for each specific matrix type.
    :return: Generated matrix.
    """
    if matrix_type == "goe":
        s = kwargs.get('c', N)
        delta = kwargs.get('delta', 1.0)
        mu = kwargs.get('mu', 0.0)
        
        mask = torch.bernoulli(torch.full((N, N), s / N))
        values = torch.normal(mu / s, delta / (2 * math.sqrt(s)), (N, N))
        matrix = values * mask
        return matrix
    
    elif matrix_type == "goe_symmetric":
        s = kwargs.get('c', N)
        delta = kwargs.get('delta', 1.0)
        mu = kwargs.get('mu', 0.0)
        
        mask = torch.bernoulli(torch.full((N, N), s / N)).triu()
        values = torch.normal(mu / s, delta / (2 * math.sqrt(s)), (N, N))
        upper_triangular = values * mask
        symmetric_matrix = upper_triangular + upper_triangular.T - torch.diag(torch.diag(upper_triangular))
        return symmetric_matrix
    
    elif matrix_type == "power_law":
        alpha = kwargs.get('alpha', 3.0)
        xmin = kwargs.get('xmin', 0.001)
        
        u = torch.rand(N, N)
        matrix = xmin * (1 - u) ** (- 1 / (alpha - 1))
        return matrix
    
    else:
        raise ValueError("Invalid matrix_type. Choose from 'goe', 'goe_symmetric', or 'power_law'.")


# Different types of input drives
def make_input_drive(N, input_type, input_norm, sigma=None):
    z = torch.full((N,), 1e-3)  # Set all elements to a small initial value
    if input_type == "localized":
        z[0] = input_norm
    elif input_type == "delocalized":
        z = torch.full((N,), 1.0)
        z = z / torch.norm(z) * input_norm
    elif input_type == "random":
        z = torch.randn(N)
        z = z / torch.norm(z) * input_norm
    elif input_type == "gaussian":
        if sigma is None:
            sigma = N / 10  # Default sigma if not provided
        z = torch.exp(-torch.arange(N).float() ** 2 / (2 * sigma ** 2))
        z = z / torch.norm(z) * input_norm
    return z

