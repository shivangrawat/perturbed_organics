import torch
import numpy as np
import math
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.signal import argrelextrema
import scipy.signal as signal


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


### ORGaNICs specific code
# Functions to sample different types of matrices

def generate_matrix(N, matrix_type, **kwargs):
    """
    Generate different types of matrices based on the matrix_type.

    :param N: Size of the matrix.
    :param matrix_type: Type of matrix to generate. Options are 'goe', 'goe_symmetric', or 'power_law'.
    :param kwargs: Additional parameters for each specific matrix type.
    :return: Generated matrix.
    """
    if matrix_type == "goe":
        s = kwargs.get("c", N)
        delta = kwargs.get("delta", 1.0)
        mu = kwargs.get("mu", 0.0)

        mask = torch.bernoulli(torch.full((N, N), s / N))
        values = torch.normal(mu / s, delta / math.sqrt(s), (N, N))
        matrix = values * mask
        return matrix

    elif matrix_type == "goe_symmetric":
        s = kwargs.get("c", N)
        delta = kwargs.get("delta", 1.0)
        mu = kwargs.get("mu", 0.0)

        mask = torch.bernoulli(torch.full((N, N), s / N))
        values = torch.normal(mu / s, delta / math.sqrt(s), (N, N))
        matrix = values * mask
        symmetric_matrix = (matrix + matrix.t()) / 2
        return symmetric_matrix

    elif matrix_type == "power_law":
        alpha = kwargs.get("alpha", 3.0)
        xmin = kwargs.get("xmin", 0.001)

        u = torch.rand(N, N)
        matrix = xmin * (1 - u) ** (-1 / (alpha - 1))
        return matrix

    else:
        raise ValueError(
            "Invalid matrix_type. Choose from 'goe', 'goe_symmetric', or 'power_law'."
        )

# Function to generate different types of input drives

def make_input_drive(N, input_type, input_norm, **kwargs):
    z = torch.full((N,), 0.0)  # Set all elements to 0
    if input_type == "localized":
        z[0] = input_norm
    elif input_type == "delocalized":
        z = torch.full((N,), 1.0)
        z = z / torch.norm(z) * input_norm
    elif input_type == "random":
        z = torch.randn(N)
        z = z / torch.norm(z) * input_norm
    elif input_type == "gaussian":
        sigma = kwargs.get("sigma", N / 10)
        z = torch.exp(-torch.arange(N).float() ** 2 / (2 * sigma**2))
        z = z / torch.norm(z) * input_norm
    return z


def nanstd(input, dim=0, keepdim=False, unbiased=True):
    """
    Compute the standard deviation of `input` along dimension `dim`, ignoring NaNs.
    
    Parameters:
      input (Tensor): The input tensor.
      dim (int or tuple of ints, optional): Dimension(s) along which to compute the std. 0 by default.
      keepdim (bool, optional): Whether the output tensor has dim retained or not.
      unbiased (bool, optional): If True, use Bessel's correction (divide by N-1).
           Otherwise, divide by N.
    
    Returns:
      Tensor: The computed standard deviation.
    """

    # Compute the mean ignoring NaNs.
    mean = torch.nanmean(input, dim=dim, keepdim=True)
    
    # Create a mask of non-NaN elements.
    mask = ~torch.isnan(input)
    
    # Compute squared differences; set differences for NaNs to 0.
    diff = (input - mean) ** 2
    diff[~mask] = 0.0

    # Sum squared differences along the specified dimension.
    sum_diff = diff.sum(dim=dim, keepdim=keepdim)
    
    # Count the number of valid (non-NaN) elements.
    count = mask.sum(dim=dim, keepdim=keepdim).to(input.dtype)
    
    # Determine the degrees of freedom correction.
    ddof = 1 if unbiased else 0
    denom = count - ddof

    # Prepare tensors for where() to handle edge cases.
    # If count == 0, return NaN.
    # If denom <= 0 (e.g. when there's only one valid element with unbiased=True), set variance to 0.
    nan_tensor = torch.full_like(sum_diff, float('nan'))
    zero_tensor = torch.zeros_like(sum_diff)
    
    variance = torch.where(
        count == 0,
        nan_tensor,
        torch.where(denom <= 0, zero_tensor, sum_diff / denom)
    )
    
    return variance.sqrt()

def is_diverging(trajectory, max_val=1e10):
    """
    Check if the trajectory is diverging (NaNs or excessively large values).
    Args:
        trajectory (torch.Tensor): Shape (time_steps, state_dim)
        max_val (float): Threshold for detecting divergence
    Returns:
        bool: True if the trajectory is diverging
    """
    return torch.any(torch.isnan(trajectory)) or torch.any(torch.abs(trajectory) > max_val)

def is_fixed_point(trajectory, last_percent=0.5, tol=1e-5):
    """
    Check if the trajectory has converged to a fixed point using the last percentage of the signal.
    
    Args:
        trajectory (torch.Tensor): Shape (time_steps, state_dim)
        last_percent (float): Fraction of the trajectory to use (0 < last_percent <= 1)
        tol (float): Tolerance for detecting fixed points
        
    Returns:
        bool: True if the trajectory is at a fixed point, False otherwise.
    """
    n_points = len(trajectory)
    if n_points == 0 or not (0 < last_percent <= 1):
        raise ValueError("last_percent must be between 0 and 1 (exclusive 0, inclusive 1) and trajectory cannot be empty.")
    
    window = int(n_points * last_percent)
    # Ensure we have at least two points to compare
    if window < 2:
        return False
    
    last_window = trajectory[-window:]  # Shape (window, state_dim)
    diffs = torch.abs(last_window[1:] - last_window[:-1])  # Shape (window-1, state_dim)
    max_diff = torch.max(diffs)
    
    return max_diff < tol

def is_periodic(signal_data, tol_period=1e-2, tolerance=1e-2):
    # Find maxima and minima
    maxima, _ = signal.find_peaks(signal_data)
    minima, _ = signal.find_peaks(-signal_data)
    
    # Combine and sort all extrema
    extrema = np.sort(np.concatenate((maxima, minima)))
    
    if len(extrema) < 4:  # Need at least two full cycles
        return False
    
    # Calculate periods
    periods = np.diff(extrema[::2])  # Every other extremum for full cycles

    mean_interval = np.mean(periods)
    if mean_interval == 0:
        return False
    rel_std_interval = np.std(periods) / mean_interval
    if rel_std_interval >= tol_period:
        return False
    
    # Check amplitudes
    amplitudes = np.abs(signal_data[extrema[1:]] - signal_data[extrema[:-1]])
    
    # Check if all amplitudes are the same
    if not np.allclose(amplitudes, np.mean(amplitudes), rtol=tolerance):
        return False
    
    return True