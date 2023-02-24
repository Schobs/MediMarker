import tensorflow as tf
from gpflow.inducing_variables import SharedIndependentInducingVariables
from gpflow.kernels import SharedIndependent, SeparateIndependent, LinearCoregionalization
from check_shapes import check_shapes
from gpflow.covariances.dispatch import Kuf


@Kuf.register(SharedIndependentInducingVariables, SeparateIndependent, object)
@check_shapes(
    "inducing_variable: [M, D, P]",
    "Xnew: [batch..., N, D2]",
    "return: [L, M, batch..., N]",
)
def Kuf_conv_shared_separate(
    inducing_variable: SharedIndependentInducingVariables,
    kernel: SeparateIndependent,
    Xnew: tf.Tensor,
) -> tf.Tensor:
    Kzxs = []
    for k in kernel.kernels:
        Xp = k.get_patches(Xnew)  # [N, num_patches, patch_len]
        bigKzx = k.base_kernel.K(
            inducing_variable.inducing_variable.Z, Xp
        )  # [M, N, P] -- thanks to broadcasting of kernels
        Kzx = tf.reduce_sum(bigKzx * k.weights if hasattr(k, "weights") else bigKzx, [2])
        Kzxs.append(Kzx / k.num_patches)
    return tf.stack(Kzxs, axis=0)


@Kuf.register(SharedIndependentInducingVariables, SharedIndependent, object)
@check_shapes(
    "inducing_variable: [M, D, P]",
    "Xnew: [batch..., N, D2]",
    "return: [M, batch..., N]",
)
def Kuf_conv_shared_shared(
    inducing_variable: SharedIndependentInducingVariables,
    kernel: SharedIndependent,
    Xnew: tf.Tensor,
) -> tf.Tensor:
    Xp = kernel.kernel.get_patches(Xnew)  # [N, num_patches, patch_len]
    bigKzx = kernel.kernel.base_kernel.K(
        inducing_variable.inducing_variable.Z, Xp
    )  # [M, N, P] -- thanks to broadcasting of kernels
    Kzx = tf.reduce_sum(bigKzx * kernel.kernel.weights if hasattr(kernel.kernel, "weights") else bigKzx, [2])
    return Kzx / kernel.kernel.num_patches


@Kuf.register(SharedIndependentInducingVariables, LinearCoregionalization, object)
@check_shapes(
    "inducing_variable: [M, D, L]",
    "kernel.W: [P, L]",
    "Xnew: [batch..., N, D2]",
    "return: [L, M, batch..., N]",
)
def Kuf_conv_shared_linear_coregionalization(
    inducing_variable: SharedIndependentInducingVariables,
    kernel: LinearCoregionalization,
    Xnew: tf.Tensor,
) -> tf.Tensor:
    Kzxs = []
    for k in kernel.kernels:
        Xp = k.get_patches(Xnew)  # [N, num_patches, patch_len]
        bigKzx = k.base_kernel.K(
            inducing_variable.inducing_variable.Z, Xp
        )  # [M, N, P] -- thanks to broadcasting of kernels
        Kzx = tf.reduce_sum(bigKzx * k.weights if hasattr(k, "weights") else bigKzx, [2])
        Kzxs.append(Kzx / k.num_patches)
    return tf.stack(Kzxs, axis=0)
