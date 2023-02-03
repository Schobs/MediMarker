from scipy.stats import zscore
import torch
import numpy as np


def normalize_cmr(image, to_tensor=False):
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

        norm_image = ((image - torch.mean(image)) / (torch.std(image) + 1e-100)).float()

    else:

        norm_image = (image - np.mean(image)) / (np.std(image) + 1e-100)

        if to_tensor:
            norm_image = torch.from_numpy(np.expand_dims(norm_image, axis=0)).float()

    return norm_image


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample["image"]
        label = sample["label"]

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
