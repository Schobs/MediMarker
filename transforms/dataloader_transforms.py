import imgaug.augmenters as iaa
import imgaug
import numpy as np
import torch
import matplotlib.pyplot as plt


def custom_flatten(iaaa_return):
    """Customn transformation to flatten an image and turn it into a tensor.

    Args:
        image (_type_): _description_
        keypoints (_type_): _description_

    Returns:
        _type_: _description_
    """
    image, keypoints = iaaa_return
    # plt.imshow(image)
    # plt.show()
    images_reshape = image[:, :]
    flattened_image = images_reshape.flatten()
    # flattened_image = flattened_image.reshape(1,flattened_image.shape[0], flattened_image.shape[1], 1)
    # flattened_image = torch.from_numpy(flattened_image)
    return flattened_image, keypoints


def get_aug_package_loader(aug_package):
    """returns a fucntion to load data augmentations from a given pakage.

    Args:
        aug_package (str): the image augmentation package to use.

    Raises:
        ValueError: if an unsupported package is given by the user.

    Returns:
        function: function to load data augmentation for the given package.
    """

    if aug_package == "imgaug":
        return get_imgaug_transforms
    else:
        raise ValueError('aug package % s not supported. Try "imgaug" '
                         % (aug_package)
                         )


def get_imgaug_transforms(data_augmentation, final_im_size):
    """Returns a data augmentation sequence from the imgaug package

    Args:
        data_augmentation (str): name of the data augmentation strategy

    Raises:
        ValueError: error if data_augmentation has not been defined as a strategy

    Returns:
        transform: sequence of transforms
    """

    if data_augmentation == "Flatten":

        def transform(image, keypoints): return custom_flatten(
            iaa.CenterCropToFixedSize(final_im_size[0], final_im_size[1])(image=image, keypoints=keypoints))

    elif data_augmentation == "AffineSimple":

        transform = iaa.Sequential(
            [
                iaa.Sometimes(
                    0.75,
                    iaa.Affine(
                        rotate=(-45, 45),
                        scale=(0.8, 1.2),
                    ),
                ),
                iaa.flip.Flipud(p=0.5),
                iaa.CenterCropToFixedSize(final_im_size[0], final_im_size[1]),
            ]
        )

    elif data_augmentation == "AffineComplex":
        transform = iaa.Sequential(
            [
                iaa.Sometimes(
                    0.5,
                    iaa.Affine(
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                        translate_percent={"x": (-0.07, 0.07), "y": (-0.07, 0.07)},
                        rotate=(-45, 45),
                        shear=(-16, 16),
                        order=[0, 1],
                    ),
                ),
                iaa.flip.Flipud(p=0.5),
                iaa.CenterCropToFixedSize(final_im_size[0], final_im_size[1]),
            ]
        )
    elif data_augmentation == "AffineComplexElastic":
        transform = iaa.Sequential(
            [
                iaa.Sometimes(
                    0.5,
                    iaa.Affine(
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                        translate_percent={"x": (-0.07, 0.07), "y": (-0.07, 0.07)},
                        rotate=(-45, 45),
                        shear=(-16, 16),
                        order=[0, 1],
                    ),
                ),
                iaa.flip.Flipud(p=0.5),
                iaa.Sometimes(
                    0.5, iaa.ElasticTransformation(alpha=(0, 200), sigma=(9, 13))
                ),
                iaa.CenterCropToFixedSize(final_im_size[0], final_im_size[1]),
            ]
        )
    elif data_augmentation == "AffineComplexElasticLight":
        transform = iaa.Sequential(
            [
                iaa.Sometimes(
                    0.5,
                    iaa.Affine(
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                        translate_percent={"x": (-0.07, 0.07), "y": (-0.07, 0.07)},
                        rotate=(-45, 45),
                        shear=(-16, 16),
                        order=[0, 1],
                    ),
                ),
                iaa.flip.Flipud(p=0.5),
                iaa.Sometimes(
                    0.5, iaa.ElasticTransformation(alpha=(0, 50), sigma=(5, 10))
                ),
                iaa.CenterCropToFixedSize(final_im_size[0], final_im_size[1]),
            ]
        )
    elif data_augmentation == "AffineComplexElasticBlur":
        transform = iaa.Sequential(
            [
                iaa.Sometimes(
                    0.5,
                    iaa.Affine(
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                        translate_percent={"x": (-0.07, 0.07), "y": (-0.07, 0.07)},
                        rotate=(-45, 45),
                        shear=(-16, 16),
                        order=[0, 1],
                    ),
                ),
                iaa.flip.Flipud(p=0.5),
                iaa.Sometimes(
                    0.5, iaa.ElasticTransformation(alpha=(0, 50), sigma=(2, 5))
                ),
                iaa.Sometimes(
                    0.5,
                    iaa.OneOf(
                        [
                            iaa.GaussianBlur((0, 0.2)),
                            iaa.AverageBlur(k=(3, 5)),
                            iaa.MedianBlur(k=(3, 5)),
                            iaa.AveragePooling(2),
                        ]
                    ),
                ),
                iaa.CenterCropToFixedSize(final_im_size[0], final_im_size[1]),
            ]
        )
    elif data_augmentation == "AffineComplexElasticBlurSharp":
        transform = iaa.Sequential(
            [
                iaa.Sometimes(
                    0.5,
                    iaa.Affine(
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                        translate_percent={"x": (-0.07, 0.07), "y": (-0.07, 0.07)},
                        rotate=(-45, 45),
                        shear=(-16, 16),
                        order=[0, 1],
                    ),
                ),
                iaa.flip.Flipud(p=0.5),
                iaa.Sometimes(
                    0.5, iaa.ElasticTransformation(alpha=(0, 50), sigma=(2, 5))
                ),
                iaa.Sometimes(
                    0.5,
                    iaa.OneOf(
                        [
                            iaa.GaussianBlur((0, 0.2)),
                            iaa.AverageBlur(k=(3, 5)),
                            iaa.MedianBlur(k=(3, 5)),
                            iaa.AveragePooling(2),
                        ]
                    ),
                ),
                iaa.Sometimes(
                    0.5,
                    iaa.SomeOf(
                        (0, 3),
                        [
                            iaa.Sharpen(alpha=(0, 0.75), lightness=(0, 0.5)),
                            iaa.Emboss(alpha=(0, 0.5), strength=(0, 1)),
                            iaa.LinearContrast((0.4, 1.6)),
                        ],
                    ),
                ),
                iaa.CenterCropToFixedSize(final_im_size[0], final_im_size[1]),
            ]
        )
    elif data_augmentation == "payer19":
        transform = iaa.Sequential(
            [
                iaa.Affine(
                    translate_px=(-20, 20),
                    rotate=(-15, 15),
                    scale=(0.6, 1.4),
                ),
                iaa.ElasticTransformation(alpha=(0, 50), sigma=(2, 5)),
                iaa.AddElementwise((-0.25, 0.25)),
                iaa.MultiplyElementwise((0.75, 1.25)),
                iaa.CenterCropToFixedSize(final_im_size[0], final_im_size[1]),
            ]
        )

    elif data_augmentation == "thaler21":
        transform = iaa.Sequential(
            [
                iaa.Affine(
                    translate_px=(-10, 10),
                    rotate=(-15, 15),
                    scale=(0.8, 1.2),
                ),
                iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                ),
                iaa.ElasticTransformation(alpha=(0, 50), sigma=(2, 5)),
                iaa.AddElementwise((-0.25, 0.25)),
                iaa.MultiplyElementwise((0.75, 1.25)),
                iaa.CenterCropToFixedSize(final_im_size[0], final_im_size[1]),
            ]
        )
    elif data_augmentation == "CenterCropOnly":
        transform = iaa.Sequential(
            [iaa.CenterCropToFixedSize(final_im_size[0], final_im_size[1])]
        )

    else:
        raise ValueError("transformations mode for dataaugmentation not recognised.")

    return transform
