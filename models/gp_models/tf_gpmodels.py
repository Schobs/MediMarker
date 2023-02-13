from __future__ import annotations
from ast import List, Tuple
from typing import Optional
import numpy as np
# import gpflow as gpf
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import gpflow as gpf
from gpflow import set_trainable
from gpflow.ci_utils import is_continuous_integration

from utils.tensorflow_utils.conv_helpers import get_inducing_patches


def affine_scalar_bijector(shift: Optional[float] = None, scale: Optional[float] = None) -> tfp.bijectors.Bijector:
    """
    Construct a bijector that applies an affine transformation to scalar inputs.

    The affine transformation is defined as y = scale * x + shift, where x is the input, 
    y is the output, and scale and shift are the parameters of the transformation.

    Args:
    shift: float, optional
        The shift parameter of the affine transformation. If None, defaults to 0.
    scale: float, optional
        The scale parameter of the affine transformation. If None, defaults to 1.

    Returns:
    tfp.bijectors.Bijector
        A bijector that applies the affine transformation to inputs.
    """
    scale_bijector = (
        tfp.bijectors.Scale(scale) if scale else tfp.bijectors.Identity()
    )
    shift_bijector = (
        tfp.bijectors.Shift(shift) if shift else tfp.bijectors.Identity()
    )
    return shift_bijector(scale_bijector)


def get_SVGP_model(num_dimensions: int, num_inducing_points: int, kern_list: List[gpf.kernels.Kernel], inducing_dist: str = 'uniform'):
    """
    This function creates a sparse variational Gaussian process (SVGP) model.

    Parameters
    ----------
    num_dimensions: int
        The number of dimensions of the input space.
    num_inducing_points: int
        The number of inducing points.
    kern_list: List[gpflow.kernels.Kernel]
        A list of kernels used to create a multi-output kernel.
    inducing_dist: str
        The distribution of inducing points. Currently, only "uniform" distribution is implemented.

    Returns
    -------
    gpflow.models.SVGP
        A sparse variational Gaussian process model.
    """

    assert inducing_dist in ["uniform"], "Only uniform inducing points are implemented"

    # Get inducing points
    if inducing_dist == "uniform":
        Zinit = np.tile(np.linspace(0, num_dimensions, num_inducing_points)
                        [:, None], num_dimensions)
    else:
        raise NotImplementedError("Only uniform inducing points are implemented")
    Z = Zinit.copy()

    # Create multi-output kernel from kernel list
    kernel = gpf.kernels.LinearCoregionalization(
        kern_list, W=np.random.randn(2, 2)
    )  # Notice that we initialise the mixing matrix W

    # create multi-output inducing variables from Z
    iv = gpf.inducing_variables.SharedIndependentInducingVariables(
        gpf.inducing_variables.InducingPoints(Z)
    )

    # initialize mean of variational posterior to be of shape MxL
    q_mu = np.zeros((num_inducing_points, 2))
    # initialize \sqrt(Σ) of variational posterior to be of shape LxMxM
    q_sqrt = np.repeat(np.eye(num_inducing_points)[None, ...], 2, axis=0) * 1.0
    # create SVGP model as usual and optimize
    return gpf.models.SVGP(
        kernel, gpf.likelihoods.Gaussian(), inducing_variable=iv, num_latent_gps=2, q_mu=q_mu, q_sqrt=q_sqrt,
    )


def get_conv_SVGP(X: List[np.ndarray], Y: List[np.ndarray], inp_dim: Tuple[int, int], num_inducing_patches: int,
                  patch_shape: List[int] = [3, 3], kern_type: str = "se") -> gpf.models.SVGP:
    """
    Returns a Gaussian process with a Convolutional kernel as the covariance function.

    Args:
    X (List[np.ndarray]): A list of 2D arrays representing the input images.
    Y (List[np.ndarray]): A list of 2D arrays representing the corresponding labels for the input images.
    inp_dim (Tuple[int, int]): The dimensions of the input images.
    num_inducing_patches (int): The number of inducing patches to use in the Gaussian process.
    patch_shape (List[int]=[3, 3]): The shape of the patches.
    kern_type (str='se'): The type of the inner kernel to use in the Convolutional kernel. 
                           Can be 'se' for SquaredExponential, 'matern52' for Matern52, or 'rbf' for RBF.

    Returns:
    gpf.models.SVGP: A Gaussian process model with a Convolutional kernel as the covariance function.

    Raises:
    ValueError: If the `kern_type` is not 'se', 'matern52', or 'rbf'.
    """

    # assert np.array([x.shape.ndims == 2 for x in X]).all(), "X must be a list of 2D arrays i.e. images"
    def f64(x): return np.array(x, dtype=np.float64)

    def positive_with_min(): return affine_scalar_bijector(shift=f64(1e-4))(
        tfp.bijectors.Softplus()
    )

    # @TOM: Should we use this for regression rather than classification? I don't rememember what you said it does.
    def constrained(): return affine_scalar_bijector(shift=f64(1e-4), scale=f64(100.0))(
        tfp.bijectors.Sigmoid()
    )

    def max_abs_1(): return affine_scalar_bijector(shift=f64(-2.0), scale=f64(4.0))(
        tfp.bijectors.Sigmoid()
    )

    if kern_type == "se":
        inner_kern = gpf.kernels.SquaredExponential()
    elif kern_type == "matern52":
        inner_kern = gpf.kernels.Matern52()
    elif kern_type == "rbf":
        inner_kern = gpf.kernels.RBF()
    else:
        raise ValueError("Invalid kernel type. Try 'se', 'matern', or 'rbf'.")

    conv_k = gpf.kernels.Convolutional(
        inner_kern, inp_dim, patch_shape
    )
    conv_k.base_kernel.lengthscales = gpf.Parameter(
        1.0, transform=positive_with_min()
    )

    # Weight scale and variance are non-identifiable. We also need to prevent variance from shooting off crazily.
    conv_k.base_kernel.variance = gpf.Parameter(1.0, transform=constrained())
    conv_k.weights = gpf.Parameter(conv_k.weights.numpy(), transform=max_abs_1())

    # TODO: Write a new get_patches function that samples patches according to a Gaussian distribution
    # centered around the label (Y) of each patch. The variance of the Gaussian should be a hyperparameter
    # we can experiment with, since we need to have some distant blank patches too.

    inducing_patches = get_inducing_patches(X, Y, inp_dim, patch_shape, num_inducing_patches, std=4)

    conv_f_i = gpf.inducing_variables.InducingPatches(inducing_patches)

    # conv_f = gpf.inducing_variables.InducingPatches(
    #     np.unique(conv_k.get_patches((np.array(X).reshape(len(X), inp_dim[0] * inp_dim[1]))).numpy().reshape(-1,
    #               (patch_shape[0]*patch_shape[1])), axis=0)[:num_inducing_patches, :]
    # )

    # @Tom how to set a mean prior as the center of the image?. Is it set q_mu to INPUT_SIZE/2?
    conv_m = gpf.models.SVGP(conv_k, gpf.likelihoods.Gaussian(), conv_f_i,  num_latent_gps=2)

    # q_mu = np.zeros((num_inducing_points, 2))
    # # initialize \sqrt(Σ) of variational posterior to be of shape LxMxM
    # q_sqrt = np.repeat(np.eye(num_inducing_points)[None, ...], 2, axis=0) * 1.0
    # create SVGP model as usual and optimize

    return conv_m
