
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


def affine_scalar_bijector(shift=None, scale=None):
    scale_bijector = (
        tfp.bijectors.Scale(scale) if scale else tfp.bijectors.Identity()
    )
    shift_bijector = (
        tfp.bijectors.Shift(shift) if shift else tfp.bijectors.Identity()
    )
    return shift_bijector(scale_bijector)


def get_SVGP_model(num_dimensions, num_inducing_points, kern_list, inducing_dist="uniform"):
    """Gets a GPflow SVGP model based on parameters

    Args:
        num_dimensions (int): Number of dimensions of the input data (flattened image size)
        num_inducing_points (int): Number of inducing points
        kern_list ([Kernels]): List of kernels to use for the multi-output kernel
        inducing_dist (str, optional): How to space the inducing points in the data. Defaults to "uniform".

    Raises:
        NotImplementedError: Only uniform inducing points are implemented

    Returns:
        gpf.models.SVGP: The SVGP model
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


def get_conv_SVGP(X, Y, inp_dim, num_inducing_patches, patch_shape=[3, 3], kern_type="se"):

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
    conv_f = gpf.inducing_variables.InducingPatches(
        np.unique(conv_k.get_patches(np.array(X)).numpy().reshape(-1,
                  (patch_shape[0]*patch_shape[1])), axis=0)[:num_inducing_patches, :]
    )

    # @Tom how to set a mean prior as the center of the image?. Is it set q_mu to INPUT_SIZE/2?
    conv_m = gpf.models.SVGP(conv_k, gpf.likelihoods.Gaussian(), conv_f,  num_latent_gps=2)

    # q_mu = np.zeros((num_inducing_points, 2))
    # # initialize \sqrt(Σ) of variational posterior to be of shape LxMxM
    # q_sqrt = np.repeat(np.eye(num_inducing_points)[None, ...], 2, axis=0) * 1.0
    # create SVGP model as usual and optimize

    return conv_m


#     def SVGP_matern(self):
#         # Get inducing points
#         if self.inducing_dist == "uniform":
#             Zinit = np.tile(np.linspace(0, self.num_dimensions, self.num_inducing_points)
#                             [:, None], self.num_dimensions)
#         else:
#             raise NotImplementedError("Only uniform inducing points are implemented")
#         Z = Zinit.copy()

#         # create multi-output kernel
#         kern_list = [
#             gpf.kernels.Matern52() + gpf.kernels.Linear() for _ in
# def SVGP_square_exponential(num_dimensions, num_inducing_points, inducing_dist="uniform"):

#     # Get inducing points
#     if inducing_dist == "uniform":
#         Zinit = np.tile(np.linspace(0, num_dimensions, num_inducing_points)[:, None], num_dimensions)
#     else:
#         raise NotImplementedError("Only uniform inducing points are implemented")
#     Z = Zinit.copy()

#     # create multi-output kernel
#     kern_list = [
#         gpf.kernels.SquaredExponential() + gpf.kernels.Linear() for _ in range(2)
#     ]
#     # Create multi-output kernel from kernel list
#     kernel = gpf.kernels.LinearCoregionalization(
#         kern_list, W=np.random.randn(2, 2)
#     )  # Notice that we initialise the mixing matrix W

#     # create multi-output inducing variables from Z
#     iv = gpf.inducing_variables.SharedIndependentInducingVariables(
#         gpf.inducing_variables.InducingPoints(Z)
#     )

#     # initialize mean of variational posterior to be of shape MxL
#     q_mu = np.zeros((num_inducing_points, 2))
#     # initialize \sqrt(Σ) of variational posterior to be of shape LxMxM
#     q_sqrt = np.repeat(np.eye(num_inducing_points)[None, ...], 2, axis=0) * 1.0
#     # create SVGP model as usual and optimize
#     return gpf.models.SVGP(
#         kernel, gpf.likelihoods.Gaussian(), inducing_variable=iv, num_latent_gps=2, q_mu=q_mu, q_sqrt=q_sqrt,
#     )


# def SVGP_matern(num_dimensions, num_inducing_points, inducing_dist="uniform"):

#     # Get inducing points
#     if inducing_dist == "uniform":
#         Zinit = np.tile(np.linspace(0, num_dimensions, num_inducing_points)[:, None], num_dimensions)
#     else:
#         raise NotImplementedError("Only uniform inducing points are implemented")
#     Z = Zinit.copy()

#     # create multi-output kernel
#     kern_list = [
#         gpf.kernels.Matern52() + gpf.kernels.Linear() for _ in range(2)
#     ]

#     # Create multi-output kernel from kernel list
#     kernel = gpf.kernels.LinearCoregionalization(
#         kern_list, W=np.random.randn(2, 2)
#     )  # Notice that we initialise the mixing matrix W

#     # create multi-output inducing variables from Z
#     iv = gpf.inducing_variables.SharedIndependentInducingVariables(
#         gpf.inducing_variables.InducingPoints(Z)
#     )

#     # initialize mean of variational posterior to be of shape MxL
#     q_mu = np.zeros((num_inducing_points, 2))
#     # initialize \sqrt(Σ) of variational posterior to be of shape LxMxM
#     q_sqrt = np.repeat(np.eye(num_inducing_points)[None, ...], 2, axis=0) * 1.0
#     # create SVGP model as usual and optimize
#     return gpf.models.SVGP(
#         kernel, gpf.likelihoods.Gaussian(), inducing_variable=iv, num_latent_gps=2, q_mu=q_mu, q_sqrt=q_sqrt,
#     )
