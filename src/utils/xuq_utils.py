import torch
import math
import numpy as np
import matplotlib.pyplot as plt

from numpy.typing import NDArray
from src.utils.utils_data_formatter import save_figure_to_image


def custom_one_hot(indices: list[int], length: int = 10):
    """
    Converts an array of indices to a 2D array of one-hot encoded vectors.

    Parameters:
    indices (list or array): Array of indices to be encoded.
    length (int): The length of the one-hot encoded vectors, default is 10.

    Returns:
    np.ndarray: 2D array of one-hot encoded vectors.
    """
    # Check for valid indices
    if any(index < 0 or index >= length for index in indices):
        raise ValueError(f"Indices are out of bounds for array of length {length}.")

    # Create a 2D array filled with zeros
    one_hot_matrix = np.zeros((len(indices), length), dtype=np.float32)

    # Set the appropriate indices to 1
    for row, index in enumerate(indices):
        one_hot_matrix[row, index] = 1.0

    return one_hot_matrix


def numpy_to_torch(tup: tuple):
    return tuple(torch.tensor(arr, dtype=torch.float32) for arr in tup)


def scale_uncertainty_attributions(uncertainty_attributions: NDArray) -> NDArray:
    """scales uncertainty attributions to sum to 100 per sample

    Args:
        uncertainty_attributions (_type_): uncertainty attributions

    Returns:
        _type_: scaled uncertainty attributions
    """
    row_sums = uncertainty_attributions.sum(axis=1, keepdims=True)
    if row_sums != 0:
        uncertainty_attributions_scaled = (uncertainty_attributions / row_sums) * 100
    else:
        uncertainty_attributions_scaled = uncertainty_attributions
    uncertainty_attributions_scaled = np.nan_to_num(uncertainty_attributions_scaled)
    return uncertainty_attributions_scaled


def gaussian_nll_loss(mu: float, sigma: float, target: torch.Tensor):
    """calculate negative log likelihood (nll) loss with gaussian density

    Args:
        mu (float): mean
        sigma (float): standard deviation
        target (torch.Tensor): torch target tensor

    Returns:
        float: nll loss
    """

    # Assuming mu is the predicted mean and sigma is the predicted variance
    nll = 0.5 * (((target - mu) ** 2) / (sigma**2) + math.log(sigma**2)).mean()
    return nll
