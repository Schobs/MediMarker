
import numpy as np
import gpflow as gpf


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


def get_conv_SVGP(X, Y, inp_dim, kern_size):
    with gpf.defer_build():
        conv_k = gpf.kernels.Convolutional(gpf.kernels.SquaredExponential(9), inp_dim, kern_size)
    conv_k.compile()
    conv_f = gpf.features.InducingPatch(np.unique(conv_k.compute_patches(X).reshape(-1, 9), axis=0))
    conv_m = gpf.models.SVGP(X, Y, conv_k, gpf.likelihoods.Gaussian(), feat=conv_f)
    conv_m.feature.trainable = False
    conv_m.kern.basekern.variance.trainable = False
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
