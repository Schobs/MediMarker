import copy
from typing import Dict, Optional, Tuple, List

import numpy as np
import tensorflow as tf
from scipy.cluster.vq import kmeans2

import gpflow as gpf
from gpflow.ci_utils import reduce_in_tests
from gpflow.utilities import to_default_float

from models.gp_models.tf_gpmodels import constrained, create_likelihood, max_abs_1, positive_with_min
from models.gp_models.unet_tf import UNetCompiled, unet_model_tf
from utils.tensorflow_utils.conv_helpers import get_inducing_patches


class KernelWithConvNN(gpf.kernels.Kernel):
    def __init__(
        self,
        image_shape: Tuple,
        output_dim: int,
        base_kernel: gpf.kernels.Kernel,
        batch_size: int,
        cnn: tf.keras.Model,
    ):
        super().__init__()
        with self.name_scope:
            self.base_kernel = base_kernel
            input_size = int(tf.reduce_prod(image_shape))
            input_shape = (input_size,)
            self.cnn = cnn

            # self.cnn = tf.keras.Sequential(
            #     [
            #         tf.keras.layers.InputLayer(
            #             input_shape=input_shape, batch_size=batch_size
            #         ),
            #         tf.keras.layers.Reshape(image_shape),
            #         tf.keras.layers.Conv2D(
            #             filters=32,
            #             kernel_size=image_shape[:-1],
            #             padding="same",
            #             activation="relu",
            #         ),
            #         tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
            #         tf.keras.layers.Conv2D(
            #             filters=64,
            #             kernel_size=(5, 5),
            #             padding="same",
            #             activation="relu",
            #         ),
            #         tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
            #         tf.keras.layers.Flatten(),
            #         tf.keras.layers.Dense(output_dim, activation="relu"),
            #         tf.keras.layers.Lambda(to_default_float),
            #     ]
            # )

            # self.cnn.build()

    def K(
        self, a_input: tf.Tensor, b_input: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        transformed_a = self.cnn(a_input)
        transformed_b = self.cnn(b_input) if b_input is not None else b_input
        return self.base_kernel.K(transformed_a, transformed_b)

    def K_diag(self, a_input: tf.Tensor) -> tf.Tensor:
        transformed_a = self.cnn(a_input)
        return self.base_kernel.K_diag(transformed_a)


class KernelSpaceInducingPoints(gpf.inducing_variables.InducingPoints):
    pass


@gpf.covariances.Kuu.register(KernelSpaceInducingPoints, KernelWithConvNN)
def Kuu(inducing_variable, kernel, jitter=0.00001):
    func = gpf.covariances.Kuu.dispatch(
        gpf.inducing_variables.InducingPoints, gpf.kernels.Kernel
    )
    return func(inducing_variable, kernel.base_kernel, jitter=jitter)


@gpf.covariances.Kuf.register(
    KernelSpaceInducingPoints, KernelWithConvNN, object
)
def Kuf(inducing_variable, kernel, a_input):
    return kernel.base_kernel(inducing_variable.Z, kernel.cnn(a_input))


def get_conv_backbone_model(X: List[np.ndarray], Y: List[np.ndarray], inp_dim: Tuple[int, int], num_inducing_points: int, batch_size: int,
                            kern_type: str = "rbf", base_kern_ls: int = 3, base_kern_var: int = 3, init_likelihood_noise: float = 1.0,
                            independent_likelihoods: bool = False, likelihood_upper_bound: float = None,
                            kl_scale: float = 1.0, ) -> gpf.models.SVGP:
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

    if kern_type == "se":
        inner_kern = gpf.kernels.SquaredExponential()
    elif kern_type == "matern12":
        inner_kern = gpf.kernels.Matern12()
    elif kern_type == "matern32":
        inner_kern = gpf.kernels.Matern32()
    elif kern_type == "rbf":
        inner_kern = gpf.kernels.RBF()
    else:
        raise ValueError("Invalid kernel type. Try 'se', 'matern', or 'rbf'.")

    # inp_dim = np.expand_dims(inp_dim, axis=0)
    kernel = KernelWithConvNN(
        inp_dim + [1],
        2,
        inner_kern,
        batch_size
    )

    inducing_variable_kmeans = kmeans2(
        np.array(X), 10, minit="points"
    )[0]

    inducing_variable_cnn = kernel.cnn(inducing_variable_kmeans)
    inducing_variable = KernelSpaceInducingPoints(inducing_variable_cnn)

    likelihood = create_likelihood(independent_likelihoods, init_likelihood_noise, likelihood_upper_bound)

    conv_m = gpf.models.SVGP(
        kernel,
        likelihood,
        inducing_variable=inducing_variable,
        num_data=len(X),
        num_latent_gps=2,
    )
    return conv_m


def get_conv_backbone_model_linear_coreg(X: List[np.ndarray], Y: List[np.ndarray], inp_dim: Tuple[int, int], num_inducing_points: int, batch_size: int,
                                         patch_shape: List[int] = [3, 3], kern_type: str = "rbf", inducing_sample_var: int = 1,
                                         base_kern_ls: int = 3, base_kern_var: int = 3, init_likelihood_noise: float = 1.0,
                                         independent_likelihoods: bool = False, likelihood_upper_bound: float = None,
                                         kl_scale: float = 1.0) -> gpf.models.SVGP:
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

    if kern_type == "se":
        inner_kern = gpf.kernels.SquaredExponential()
    elif kern_type == "matern12":
        inner_kern = gpf.kernels.Matern12()
    elif kern_type == "matern32":
        inner_kern = gpf.kernels.Matern32()
    elif kern_type == "rbf":
        inner_kern = gpf.kernels.RBF()
    else:
        raise ValueError("Invalid kernel type. Try 'se', 'matern', or 'rbf'.")

    image_shape = inp_dim + [1]
    input_size = int(tf.reduce_prod(image_shape))
    input_shape = (input_size,)
    # inp_dim = np.expand_dims(inp_dim, axis=0)
    cnn = UNetCompiled(1, input_shape=input_shape, image_shape=image_shape,
                       out_dim=64, batch_size=batch_size)
    convs_k = []
    for i in range((2)):

        conv_k = gpf.kernels.Convolutional(
            inner_kern, inp_dim, patch_shape
        )
        conv_k.base_kernel.lengthscales = gpf.Parameter(
            [base_kern_ls] * int(patch_shape[0]*patch_shape[1]), transform=positive_with_min()
        )

        # Weight scale and variance are non-identifiable. We also need to prevent variance from shooting off crazily.
        conv_k.base_kernel.variance = gpf.Parameter(base_kern_var, transform=constrained())
        conv_k.weights = gpf.Parameter(conv_k.weights.numpy(), transform=max_abs_1())

        kernel = KernelWithConvNN(
            inp_dim + [1],
            2,
            conv_k,
            batch_size,
            cnn=cnn
        )

        convs_k.append(kernel)

    kernel = gpf.kernels.LinearCoregionalization(
        convs_k, W=np.random.randn(2, 2)
    )

    inducing_patches = get_inducing_patches(X, Y, inp_dim, inp_dim, num_inducing_points, std=1)
    # inducing_patches = inducing_patches.reshape(-1, inp_dim[0], inp_dim[1], 1)
    # inducing_variable_kmeans = kmeans2(
    #     np.array(X), 10, minit="points"
    # )[0]

    inducing_variable_cnn = kernel.kernels[0].cnn(inducing_patches)

    ivc_flat = inducing_variable_cnn.numpy().reshape(-1, patch_shape[0] * patch_shape[1])
    conv_f_i = gpf.inducing_variables.InducingPatches(ivc_flat)

    iv = gpf.inducing_variables.SharedIndependentInducingVariables(
        conv_f_i
    )

    # inducing_variable = KernelSpaceInducingPoints(inducing_variable_cnn)

    # iv = gpf.inducing_variables.SharedIndependentInducingVariables(
    #     inducing_variable
    # )
    likelihood = create_likelihood(independent_likelihoods, init_likelihood_noise, likelihood_upper_bound)

    conv_m = gpf.models.SVGP(
        kernel,
        likelihood,
        inducing_variable=iv,
        num_data=len(X),
        num_latent_gps=2,
    )
    return conv_m
