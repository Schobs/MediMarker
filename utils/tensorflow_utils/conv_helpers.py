from ast import List, Tuple
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def get_inducing_patches(images: List, landmarks: List, original_image_dims: List, patch_shape: Tuple,
                         num_inducing_patches: int, std: int = 4) -> List:
    """
    This function generates patches from a list of images according to a Gaussian distribution centered on the landmarks.

    Parameters:
    images (List[np.ndarray]): A list of images.
    landmarks (List[Tuple[int, int]]): A list of 2D landmarks, each relating to each image.
    original_image_dims (List[int]): The original dimensions of the images.
    patch_shape (Tuple[int, int]): The shape of the patches to be extracted.
    num_inducing_patches (int): The number of patches to be extracted.
    std (int, optional): The standard deviation of the Gaussian distribution. Default is 4.

    Returns:
    List[np.ndarray]: A list of extracted patches.
    """

    patches_per_image = max(1, num_inducing_patches // len(images)) + 1

    minus_mean_x = patch_shape[0] // 2
    plus_mean_x = patch_shape[0] - minus_mean_x

    minus_mean_y = patch_shape[1] // 2
    plus_mean_y = patch_shape[1] - minus_mean_y

    all_patches = []
    for i, (img, lm) in enumerate(zip(images, landmarks)):
        img = tf.reshape(img, original_image_dims)
        h, w = img.shape[:2]
        patches = []
        for j in range(patches_per_image):
            x = int(np.random.normal(lm[0], std))
            y = int(np.random.normal(lm[1], std))
            x = max(minus_mean_x, min(x, w - plus_mean_x))
            y = max(minus_mean_y, min(y, h - plus_mean_y))
            patch = img[y-minus_mean_y:y+plus_mean_y, x-minus_mean_x:x+plus_mean_x]
            patches.append(tf.reshape(patch, [-1]))
        all_patches.append(patches)
    return_patches = tf.stack(all_patches)

    inducing_patches_flattened = return_patches.numpy().reshape(-1, patch_shape[0] * patch_shape[1])
    inducing_patches_randomly_sampled = inducing_patches_flattened[np.random.choice(
        inducing_patches_flattened.shape[0], num_inducing_patches, replace=False)]

    return inducing_patches_randomly_sampled
