from scipy.stats import zscore
import torch
import numpy as np
from albumentations.augmentations.transforms import ImageOnlyTransform
from albumentations.core.transforms_interface import BasicTransform
import matplotlib.pyplot as plt

import warnings

def normalize_cmr(image,to_tensor=False):
    """Adds small epsilon to std to avoid divide by zero

    Args:
        image (_type_): _description_
        to_tensor (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    # warnings.filterwarnings("error")
    # image = sample["image"]
    if torch.is_tensor(image):
        
        norm_image = ((image-torch.mean(image))/(torch.std(image)+1e-100)).float()
       
    else:

        norm_image = ((image-np.mean(image))/(np.std(image)+1e-100))
    
        if to_tensor:
            norm_image = torch.from_numpy(np.expand_dims(norm_image, axis=0)).float()

    return norm_image


class NormalizeZScore(ImageOnlyTransform):
    def __init__(
        self,
        always_apply=True,
        p=1.0
    ):
        super(NormalizeZScore, self).__init__()

    def apply(self, image, **params):
        print("applying normalization")
        return (zscore(image, axis=None))
        # norm_image = (image-np.mean(image))/np.std(image)
        # return norm_image


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        

        # print("before image shape: ", image.shape)
        # image = image.transpose((2, 0, 1))
        sample["image"] = torch.from_numpy(image).float()

        all_seg_labels = []
        for maps in sample["label"]:
            all_seg_labels.append(torch.from_numpy(maps).float())

        sample["label"] = all_seg_labels

        return sample


# class ToTensorImgaug(object):
#     """Convert ndarrays in sample to Tensors."""

#     def __call__(self, sample):
#         image = torch.from_numpy(sample.image)
#         coords = torch.from_numpy([sample.keypoints.x_int, sample.keypoints.y_int])

#         sample["image"] = image

#         sample["target_coords"] = coords


#     def func_images(images, random_state, parents, hooks):
#         bg_ids = random_state.randint(0, len(backgrounds), size=(len(images),))
#         result = []
#         for image, bg_id in zip(images, bg_ids):
#             image_small = ia.imresize_single_image(image, (64, 64))
#             image_aug = np.copy(backgrounds[bg_id])
#             image_aug[32:-32, 32:-32, :] = image_small
#             result.append(image_aug)
#         return result
        
#     def func_heatmaps(heatmaps, random_state, parents, hooks):
#         return heatmaps
        
#     def func_keypoints(keypoints_on_images, random_state, parents, hooks):
#         return keypoints_on_images

# bg_augmenter = iaa.Lambda(
#     func_images=func_images,
#     func_heatmaps=func_heatmaps,
#     func_keypoints=func_keypoints
# )

class HeatmapsToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, heatmaps):


        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        

        # print("before image shape: ", image.shape)
        # image = image.transpose((2, 0, 1))

        all_seg_labels = []
        for maps in heatmaps:
            all_seg_labels.append(torch.from_numpy(maps).float())


        return all_seg_labels


class ToTensorV3(BasicTransform):
    """Convert image and mask to `torch.Tensor`. The numpy `HWC` image is converted to pytorch `CHW` tensor.
    If the image is in `HW` format (grayscale image), it will be converted to pytorch `HW` tensor.
    This is a simplified and improved version of the old `ToTensor`
    transform (`ToTensor` was deprecated, and now it is not present in Albumentations. You should use `ToTensorV2`
    instead).

    Args:
        transpose_mask (bool): if True and an input mask has three dimensions, this transform will transpose dimensions
        so the shape `[height, width, num_channels]` becomes `[num_channels, height, width]`. The latter format is a
        standard format for PyTorch Tensors. Default: False.
    """

    def __init__(self, transpose_mask=False, always_apply=True, p=1.0):
        super(ToTensorV3, self).__init__(always_apply=always_apply, p=p)
        self.transpose_mask = transpose_mask

    @property
    def targets(self):
        return {"image": self.apply, "mask": self.apply_to_mask}

    def apply(self, img, **params):  # skipcq: PYL-W0613
        if len(img.shape) not in [2, 3]:
            raise ValueError("Albumentations only supports images in HW or HWC format")

        if len(img.shape) == 2:
            img = np.expand_dims(img, 2)

        return torch.from_numpy(img.transpose(2, 0, 1)).float()

    def apply_to_mask(self, mask, **params):  # skipcq: PYL-W0613
        if self.transpose_mask and mask.ndim == 3:
            mask = mask.transpose(2, 0, 1)
        return torch.from_numpy(mask).float()

    def get_transform_init_args_names(self):
        return ("transpose_mask",)

    def get_params_dependent_on_targets(self, params):
        return {}