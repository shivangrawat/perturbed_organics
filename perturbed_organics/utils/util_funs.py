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


def make_spike_train(simulation, dt=1e-4, dtype=torch.float64):
    """
    This function receives an inhomogeneous firing rate as a function of time and dt.
    We approximate the Poisson process by a Bernoulli process.
    
    :param simulation: A torch.Tensor containing the firing rate.
    :param dt: Time step.
    :param dtype: Data type to use (default: torch.float64).
    :return: The spike train as a boolean tensor.
    """
    simulation = simulation.to(dtype)
    uniform_rand = torch.rand(simulation.shape, dtype=dtype, device=simulation.device)
    spike_prob = simulation * dt
    spike_train = uniform_rand < spike_prob
    return spike_train


def generate_stable_system(n, dtype=torch.float64):
    """
    Returns a randomly generated stable linear dynamical system (with eigenvalues whose real parts are negative).

    :param n: Dimension of the system.
    :param dtype: Data type to use (default: torch.float64).
    :return: A stable system matrix.
    """
    A = torch.randn(n, n, dtype=dtype)
    L, V = torch.linalg.eig(A)
    L = -torch.abs(torch.real(L)) + 1j * torch.imag(L)
    return torch.real(V @ torch.diag(L) @ torch.linalg.inv(V))


def cholesky_decomposition(A, dtype=torch.float64):
    """
    Returns the Cholesky decomposition of a matrix A.

    :param A: The matrix to decompose.
    :param dtype: Data type to use (default: torch.float64).
    :return: The lower-triangular Cholesky factor.
    """
    A = A.to(dtype)
    L = torch.linalg.cholesky(A)
    return L


### ORGaNICs specific code
# Functions to sample different types of matrices

def generate_matrix(N, matrix_type, dtype=torch.float64, **kwargs):
    """
    Generate different types of matrices based on the matrix_type.

    :param N: Size of the matrix.
    :param matrix_type: Type of matrix to generate. Options are 'goe', 'goe_symmetric', or 'power_law'.
    :param dtype: Data type to use (default: torch.float64).
    :param kwargs: Additional parameters for each specific matrix type.
    :return: Generated matrix.
    """
    if matrix_type == "goe":
        s = kwargs.get("c", N)
        delta = kwargs.get("delta", 1.0)
        mu = kwargs.get("mu", 0.0)

        mask = torch.bernoulli(torch.full((N, N), s / N, dtype=dtype))
        values = torch.normal(mu / s, delta / math.sqrt(s), (N, N)).to(dtype)
        matrix = values * mask
        return matrix

    elif matrix_type == "goe_symmetric":
        s = kwargs.get("c", N)
        delta = kwargs.get("delta", 1.0)
        mu = kwargs.get("mu", 0.0)

        mask = torch.bernoulli(torch.full((N, N), s / N, dtype=dtype))
        values = torch.normal(mu / s, delta / math.sqrt(s), (N, N)).to(dtype)
        matrix = values * mask
        symmetric_matrix = (matrix + matrix.t()) / 2
        return symmetric_matrix

    elif matrix_type == "power_law":
        alpha = kwargs.get("alpha", 3.0)
        xmin = kwargs.get("xmin", 0.001)

        u = torch.rand(N, N, dtype=dtype)
        matrix = xmin * (1 - u) ** (-1 / (alpha - 1))
        return matrix

    else:
        raise ValueError(
            "Invalid matrix_type. Choose from 'goe', 'goe_symmetric', or 'power_law'."
        )


def make_input_drive(N, input_type, input_norm, dtype=torch.float64, **kwargs):
    """
    Generate an input drive vector.

    :param N: Size of the input vector.
    :param input_type: Type of input drive ('localized', 'delocalized', 'random', 'gaussian').
    :param input_norm: Normalization constant for the input drive.
    :param dtype: Data type to use (default: torch.float64).
    :param kwargs: Additional parameters for specific input types.
    :return: Input drive vector.
    """
    z = torch.full((N,), 0.0, dtype=dtype)
    if input_type == "localized":
        z[0] = input_norm
    elif input_type == "delocalized":
        z = torch.full((N,), 1.0, dtype=dtype)
        z = z / torch.norm(z) * input_norm
    elif input_type == "random":
        z = torch.randn(N, dtype=dtype)
        z = z / torch.norm(z) * input_norm
    elif input_type == "gaussian":
        sigma = kwargs.get("sigma", N / 10)
        z = torch.exp(-torch.arange(N, dtype=dtype) ** 2 / (2 * sigma**2))
        z = z / torch.norm(z) * input_norm
    return z


def nanstd(input, dim=0, keepdim=False, unbiased=True):
    """
    Compute the standard deviation of `input` along dimension `dim`, ignoring NaNs.
    
    Parameters:
      input (Tensor): The input tensor.
      dim (int or tuple of ints, optional): Dimension(s) along which to compute the std. 0 by default.
      keepdim (bool, optional): Whether the output tensor has dim retained.
      unbiased (bool, optional): If True, use Bessel's correction (divide by N-1); otherwise, divide by N.
    
    Returns:
      Tensor: The computed standard deviation.
    """
    mean = torch.nanmean(input, dim=dim, keepdim=True)
    mask = ~torch.isnan(input)
    diff = (input - mean) ** 2
    diff[~mask] = 0.0
    sum_diff = diff.sum(dim=dim, keepdim=keepdim)
    count = mask.sum(dim=dim, keepdim=keepdim).to(input.dtype)
    ddof = 1 if unbiased else 0
    denom = count - ddof
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
    Check if the trajectory is diverging (contains NaNs or excessively large values).

    :param trajectory: Tensor with shape (time_steps, state_dim).
    :param max_val: Threshold for detecting divergence.
    :return: True if diverging; False otherwise.
    """
    return torch.any(torch.isnan(trajectory)) or torch.any(torch.abs(trajectory) > max_val)


def is_fixed_point(trajectory, last_percent=0.5, tol=1e-6):
    """
    Check if the trajectory has converged to a fixed point using the last percentage of the signal.
    
    :param trajectory: Tensor with shape (time_steps, state_dim).
    :param last_percent: Fraction of the trajectory to use (0 < last_percent <= 1).
    :param tol: Tolerance for detecting fixed points.
    :return: True if the trajectory is at a fixed point; False otherwise.
    """
    n_points = len(trajectory)
    if n_points == 0 or not (0 < last_percent <= 1):
        raise ValueError("last_percent must be between 0 and 1 (exclusive 0, inclusive 1) and trajectory cannot be empty.")
    
    window = int(n_points * last_percent)
    if window < 2:
        return False
    
    last_window = trajectory[-window:]
    diffs = torch.abs(last_window[1:] - last_window[:-1])
    max_diff = torch.max(diffs)
    
    return max_diff < tol


def is_periodic(signal_data, tol_period=1e-2, tolerance=1e-2):
    """
    Check if a signal is periodic based on its extrema.
    
    :param signal_data: 1D array or list of signal values.
    :param tol_period: Relative tolerance for the period consistency.
    :param tolerance: Relative tolerance for amplitude consistency.
    :return: True if the signal is periodic; False otherwise.
    """
    maxima, _ = signal.find_peaks(signal_data)
    minima, _ = signal.find_peaks(-signal_data)
    extrema = np.sort(np.concatenate((maxima, minima)))
    
    if len(extrema) < 4:  # Need at least two full cycles
        return False
    
    periods = np.diff(extrema[::2])
    mean_interval = np.mean(periods)
    if mean_interval == 0:
        return False
    rel_std_interval = np.std(periods) / mean_interval
    if rel_std_interval >= tol_period:
        return False
    
    amplitudes = np.abs(signal_data[extrema[1:]] - signal_data[extrema[:-1]])
    if not np.allclose(amplitudes, np.mean(amplitudes), rtol=tolerance):
        return False
    
    return True
