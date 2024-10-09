import torch
import numpy as np
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

