import os
from pathlib import Path

import numpy as np
import torch
from imgaug.augmentables import Keypoint, KeypointsOnImage
from torch.utils import data
from torchvision import transforms
from transforms.transformations import (
    HeatmapsToTensor,
    normalize_cmr,
)
from transforms.dataloader_transforms import get_aug_package_loader

from utils.data.load_data import get_datatype_load, load_aspire_datalist, load_and_resize_image
from utils.im_utils.visualisation import visualize_patch


from utils.logging.python_logger import get_logger

from abc import ABC, ABCMeta


class DatasetMeta(ABCMeta, type(data.Dataset)):
    pass


class DatasetBase(ABC, metaclass=DatasetMeta):
    """
    A custom dataset superclass for loading landmark localization data

    Args:
        name (str): Dataset name.
        split (str): Data split type (train, valid or test).
        image_path (str): local directory of image path (default: "./data").
        annotation_path (str): local directory to file path with annotations.
        annotation_set (str): which set of annotations to use [junior, senior, challenge] (default: "junior")
        image_modality (str): Modality of image (default: "CMRI").


    References:
        #TO DO
    """

    def __init__(
        self,
        landmarks,
        sigmas,
        LabelGenerator,
        hm_lambda_scale: float,
        annotation_path: str,
        generate_hms_here: bool,
        sample_mode: str,
        image_modality: str = "CMRI",
        split: str = "training",
        root_path: str = "./data",
        cv: int = -1,
        cache_data: bool = False,
        debug: bool = False,
        input_size=[512, 512],
        sample_patch_size=[512, 512],
        sample_patch_bias=0.66,
        sample_patch_from_resolution=[512, 512],
        num_res_supervisions: int = 5,
        data_augmentation_strategy: str = None,
        data_augmentation_package: str = None,
        dataset_split_size: int = -1,
        additional_sample_attribute_keys=[],

    ):
        """Initialize the dataset. This is the base class for all datasets.

        Args:
            landmarks (_type_): _description_
            sigmas (_type_): _description_
            LabelGenerator (_type_): _description_
            hm_lambda_scale (float): _description_
            annotation_path (str): _description_
            generate_hms_here (bool): _description_
            sample_mode (str): _description_
            image_modality (str, optional): _description_. Defaults to "CMRI".
            split (str, optional): _description_. Defaults to "training".
            root_path (str, optional): _description_. Defaults to "./data".
            cv (int, optional): _description_. Defaults to -1.
            cache_data (bool, optional): _description_. Defaults to False.
            debug (bool, optional): _description_. Defaults to False.
            input_size (list, optional): _description_. Defaults to [512, 512].
            sample_patch_size (list, optional): _description_. Defaults to [512, 512].
            sample_patch_bias (float, optional): _description_. Defaults to 0.66.
            sample_patch_from_resolution (list, optional): _description_. Defaults to [512, 512].
            num_res_supervisions (int, optional): _description_. Defaults to 5.
            data_augmentation_strategy (str, optional): _description_. Defaults to None.
            data_augmentation_package (str, optional): _description_. Defaults to None.
            dataset_split_size (int, optional): _description_. Defaults to -1.
            additional_sample_attribute_keys (list, optional): _description_. Defaults to [].

        Raises:
            ValueError: _description_
            ValueError: _description_
        """

        super(DatasetBase, self).__init__()

        # Logger
        self.logger = get_logger("root")

        # We are passing in the label generator here, this is unique to each model_trainer class.
        self.LabelGenerator = LabelGenerator

        self.data_augmentation_strategy = data_augmentation_strategy
        self.hm_lambda_scale = hm_lambda_scale
        self.data_augmentation_package = data_augmentation_package

        self.generate_hms_here = generate_hms_here
        self.sample_mode = sample_mode
        self.sample_patch_size = sample_patch_size
        self.sample_patch_bias = sample_patch_bias
        self.sample_patch_from_resolution = sample_patch_from_resolution

        self.dataset_split_size = dataset_split_size

        # Additional sample attributes found in the json datalist to return with each sample
        self.additional_sample_attribute_keys = additional_sample_attribute_keys
        self.additional_sample_attributes = {
            k: [] for k in self.additional_sample_attribute_keys
        }

        if self.sample_mode == "patch":
            # Get the patches origin information. Use this for stitching together in valid/testing
            self.load_im_size = sample_patch_from_resolution
            self.input_size = sample_patch_size

            # Coordinates of each patch origin in the image
            self.patchified_idxs = self.get_patch_stitching_info(
                self.load_im_size, self.sample_patch_size
            )

            # if we're sampling patches w/o aug we still need to center crop so change the aug strategy
            if self.data_augmentation_strategy == None:
                self.data_augmentation_package = "imgaug"
                self.data_augmentation_strategy = "CenterCropOnly"

        elif self.sample_mode == "full":
            #  If not sample_patches then just 1 big patch!
            self.patchified_idxs = [[0, 0]]
            self.load_im_size = input_size
            self.input_size = input_size
        else:
            raise ValueError("sample mode %s not recognized." % self.sample_mode)

        self.heatmap_label_size = self.input_size
        # ratio between original image and load_image size, to resize landmarks.

        if self.data_augmentation_strategy == None:
            self.logger.info("No data Augmentation for %s split.", split)
        else:
            # Get data augmentor for the correct package
            self.aug_package_loader = get_aug_package_loader(
                self.data_augmentation_package
            )
            # Get specific data augmentation strategy
            self.transform = self.aug_package_loader(
                self.data_augmentation_strategy, self.input_size
            )
            self.logger.info(
                "Using data augmentation package %s and strategy %s for %s split.",
                data_augmentation_package, self.data_augmentation_strategy, split
            )

        self.heatmaps_to_tensor = transforms.Compose([HeatmapsToTensor()])

        self.root_path = Path(root_path)
        self.annotation_path = Path(annotation_path)
        self.image_modality = image_modality
        self.landmarks = landmarks
        self.split = split
        self.sigmas = sigmas

        self.cv = cv
        self.cache_data = cache_data
        self.debug = debug

        self.num_res_supervisions = num_res_supervisions

        # Lists to save the image paths (or images if caching), target coordinates (scaled to input size), and full resolution coords.
        self.images = []
        self.target_coordinates = []
        self.full_res_coordinates = []  # full_res will be same as target if input and original image same size
        self.image_paths = []
        self.uids = []
        self.annotation_available = []
        self.original_image_sizes = []
        self.image_resizing_factors = []

        # We are using cross-validation, following our convention of naming each json train with the append "foldX" where (X= self.cv)
        if cv >= 0:
            label_std = os.path.join("fold" + str(self.cv) + ".json")
            self.logger.info(
                "Loading %s data for CV %s ", self.split, os.path.join(self.annotation_path, label_std)
            )
            datalist = load_aspire_datalist(
                os.path.join(self.annotation_path, label_std),
                data_list_key=self.split,
                base_dir=self.root_path,
            )

            if self.dataset_split_size != -1:
                datalist = datalist[: self.dataset_split_size]
                self.logger.info("datalist: %s", datalist)
        # Not using CV, load the specified json file
        else:
            self.logger.info(
                "Loading %s data (no CV) for %s ", self.split, self.annotation_path
            )
            datalist = load_aspire_datalist(
                self.annotation_path, data_list_key=self.split, base_dir=self.root_path
            )

        # datalist = datalist[:20]

        # based on first image extenstion, get the load function.
        self.datatype_load = get_datatype_load(datalist[0]["image"])

        self.load_function = lambda img: img

        for idx, data in enumerate(datalist):
            # Add coordinate labels as sample attribute, if annotations available
            if (not isinstance(data["coordinates"], list)) or (
                "has_annotation" in data.keys() and data["has_annotation"] == False
            ):
                # Case when data has no annotation, i.e. inference only, just set target coords to 0,0 and annotation_available to False
                interested_landmarks = np.array([[0, 0]] * len(self.landmarks))
                self.annotation_available.append(False)
                self.full_res_coordinates.append(interested_landmarks)

                if self.split == "training" or self.split == "validation":
                    raise ValueError(
                        "Training/Validation data must have annotations. Check your data. Sample that failed: ",
                        data,
                    )
            else:
                # Case when we have annotations.
                interested_landmarks = np.array(data["coordinates"])[self.landmarks, :2]
                self.full_res_coordinates.append(np.array(data["coordinates"])[self.landmarks, :2])
                self.annotation_available.append(True)

            if self.cache_data:
                # Determine original size and log whether we needed to resize it
                (
                    resized_factor,
                    original_size,
                    image,
                    interested_landmarks,
                ) = load_and_resize_image(data["image"], interested_landmarks, self.load_im_size, self.datatype_load)
                self.images.append(image)
                self.image_resizing_factors.append(resized_factor)
                self.original_image_sizes.append(original_size)

            else:
                # Not caching, so just append image path.
                self.images.append(data["image"])

            self.target_coordinates.append(interested_landmarks)
            self.image_paths.append(data["image"])
            self.uids.append(data["id"])

            # Extended dataset class can add more attributes to each sample here
            self.add_additional_sample_attributes(data)

        if self.cache_data:
            self.logger.info(
                "Cached all %s data in memory. Length of %s", self.split,  len(self.images)
            )
        else:
            self.logger.info(
                "Not caching %s image data in memory, will load on the fly. Length of %s", self.split, len(self.images)
            )

        self.check_uids_unique()

    def __len__(self):
        return len(self.images)

    def set_use_cache(self, use_cache):
        self.use_cache = use_cache

    def check_uids_unique(self):
        """Make sure all uids are unique
        """
        non_unique = [
            [x, self.image_paths[x_idx]]
            for x_idx, x in enumerate(self.uids)
            if self.uids.count(x) > 1
        ]
        assert len(non_unique) == 0, (
            f"Not all uids are unique! Check your data. {len(non_unique)} non-unqiue uids from {len(self.uids)} samples , they are: {non_unique}"
        )

    def add_additional_sample_attributes(self, extra_data):
        """
        Add more attributes to each sample.

        """
        for k_ in self.additional_sample_attribute_keys:
            keyed_data = extra_data[k_]
            self.additional_sample_attributes[k_].append(keyed_data)

    def __getitem__(self, index):
        """Main function of the dataloader. Gets a data sample.



        Args:
            index (_type_): _description_

        Returns:
            It must return a dictionary with the keys:
            sample = {
                "image" (torch.tensor, shape (1, H, W)): tensor of input image to network.
                "label" (Dict of torch.tensors):  if self.generate_hms_here bool -> (Dictionary of labels e.g. tensor heatmaps, see LabelGenerator for details); else -> [].
                "target_coords (np.array, shape (num_landmarks, 2))": list of target coords of for landmarks, same scale as input to model.
                "landmarks_in_indicator" (list of 1s and/or 0s, shape (1, num_landmarks)): list of bools, whether the landmark is in the image or not.
                "full_res_coords" (np.array, shape (num_landmarks, 2)): original list of coordinates of shape ,same scale as original image (so may be same as target_coords)
                "image_path" (str): path to image, from the JSON file.
                "uid" (str): sample's unique id, from the JSON file.
                "annotation_available" (bool): Whether the JSON file provided annotations for this sample (necessary for training and validation).
                "resizing_factor" (np.array, shape (1,2)): The x and y scale factors that the image was resized by to fit the network input size.
                "original_image_size" (np.array, shape (2,1)): The resolution of the original image before it was resized.
                ANY EXTRA ATTRIBUTES ADDED BY add_additional_sample_attributes()
             }


        """

        hm_sigmas = self.sigmas
        coords = self.target_coordinates[index]
        full_res_coods = self.full_res_coordinates[index]
        im_path = self.image_paths[index]
        run_time_debug = False
        this_uid = self.uids[index]
        is_annotation_available = self.annotation_available[index]
        x_y_corner = [0, 0]
        image = self.load_function(self.images[index])

        # If we cached the data, we don't need to get original image size. If not, we need to load it here.
        if self.cache_data:
            resized_factor = self.image_resizing_factors[index]
            original_size = self.original_image_sizes[index]

        else:
            resized_factor, original_size, image, coords = load_and_resize_image(
                image, coords, self.load_im_size, self.datatype_load
            )

        untransformed_coords = coords
        untransformed_im = image

        # Do data augmentation
        if self.data_augmentation_strategy != None:

            # By default, the origin is 0,0 unless we sample from the middle of the image somewhere.
            # If sampling patches we first sample the patch with a little wiggle room, & normalize the lms. The transform center-crops it back.
            if self.sample_mode == "patch":
                (
                    untransformed_im,
                    untransformed_coords,
                    landmarks_in_indicator,
                    x_y_corner,
                ) = self.sample_patch(untransformed_im, untransformed_coords)

            kps = KeypointsOnImage(
                [Keypoint(x=coo[0], y=coo[1]) for coo in untransformed_coords],
                shape=untransformed_im[0].shape,
            )

            transformed_sample = self.transform(
                image=untransformed_im[0], keypoints=kps
            )  # list where [0] is image and [1] are coords.
            input_image = normalize_cmr(transformed_sample[0], to_tensor=True)
            input_coords = np.array([[coo.x, coo.y] for coo in transformed_sample[1]])

            # Recalculate indicators incase transform pushed out/in coords.
            landmarks_in_indicator = [
                1
                if (
                    (0 <= xy[0] <= self.input_size[0])
                    and (0 <= xy[1] <= self.input_size[1])
                )
                else 0
                for xy in input_coords
            ]

        else:

            input_coords = coords
            input_image = torch.from_numpy(image).float()
            landmarks_in_indicator = [1 for xy in input_coords]

        if self.generate_hms_here:

            label = self.LabelGenerator.generate_labels(
                input_coords,
                x_y_corner,
                landmarks_in_indicator,
                self.heatmap_label_size,
                hm_sigmas,
                self.num_res_supervisions,
                self.hm_lambda_scale,
            )
        else:
            label = []

            # If coordinates are cutoff by augmentation throw a run time error.
            # if len(np.array(input_coords)) <len(coords) or (len([n for n in (input_coords).flatten() if n < 0])>0) :
            #     print("input coords: ", input_coords)
            #     print("some coords have been cut off! You need to change the data augmentation, it's too strong.")
            # run_time_debug = True
        # else:
        #     print("ok")

        sample = {
            "image": input_image,
            "label": label,
            "target_coords": input_coords,
            "landmarks_in_indicator": landmarks_in_indicator,
            "full_res_coords": full_res_coods,
            "image_path": im_path,
            "uid": this_uid,
            "annotation_available": is_annotation_available,
            "resizing_factor": resized_factor,
            "original_image_size": original_size,
        }

        # add additional sample attributes from child class.
        for key_ in list(self.additional_sample_attributes.keys()):
            sample[key_] = self.additional_sample_attributes[key_][index]

        if self.debug or run_time_debug:
            self.logger.info("sample: %s", sample)
            self.LabelGenerator.debug_sample(
                sample, untransformed_im, untransformed_coords
            )
        return sample

    def generate_labels(self, landmarks, sigmas):
        """Generate heatmap labels using same method as in _get_item__.
        Note, this does not work with patch-based yet (note the x_y_corner is hard coded as [0,0]. This is raised in argument checkers.)

        Args:
            landmarks (_type_): _description_
            sigmas (_type_): _description_

        Returns:

        """

        # landmarks_in_indicator = [1 for xy in landmarks  ]
        landmarks_in_indicator = [
            1
            if (
                (0 <= xy[0] <= self.input_size[0])
                and (0 <= xy[1] <= self.input_size[1])
            )
            else 0
            for xy in landmarks
        ]

        x_y_corner = [0, 0]

        return self.LabelGenerator.generate_labels(
            landmarks,
            x_y_corner,
            landmarks_in_indicator,
            self.input_size,
            sigmas,
            self.num_res_supervisions,
            self.hm_lambda_scale,
        )

    def sample_patch(self, image, landmarks, lm_safe_region=0, safe_padding=128):
        """Samples a patch from the image. It ensures a landmark is in a patch with a self.sample_patch_bias% chance.
            The patch image is larger than the patch-size by safe_padding on every side for safer data augmentation.
            Therefore, the image is first padded with zeros on each side to stop out of bounds when sampling from the edges.

        Args:
            image (_type_): image to sample
            landmarks (_type_): list of landmarks
            lm_safe_region (int, optional): # pixels away from the edge the landmark must be to count as "in" the patch . Defaults to 0.
            safe_padding (int, optional): How much bigger on each edge the patch should be for safer data augmentation . Defaults to 128.

        Returns:
            _type_: cropped padded sample
            landmarks normalised to within the patch
            binary indicator of which landmarks are in the patch.

        """

        z_rand = np.random.uniform(0, 1)
        landmarks_in_indicator = []
        if z_rand >= (1 - self.sample_patch_bias):

            # Keep sampling until landmark is in patch
            while 1 not in landmarks_in_indicator:
                landmarks_in_indicator = []

                #
                y_rand = np.random.randint(
                    0, self.load_im_size[1] - self.sample_patch_size[1]
                )
                x_rand = np.random.randint(
                    0, self.load_im_size[0] - self.sample_patch_size[0]
                )

                for lm in landmarks:
                    landmark_in = 0

                    # Safe region means landmark is not right on the edge
                    if (
                        y_rand + lm_safe_region
                        <= lm[1]
                        <= (y_rand + self.sample_patch_size[1]) - lm_safe_region
                    ):
                        if (
                            x_rand + lm_safe_region
                            <= lm[0]
                            <= (x_rand + self.sample_patch_size[0]) - lm_safe_region
                        ):
                            landmark_in = 1

                    landmarks_in_indicator.append(landmark_in)

                # Tested with the extremes, its all ok.
                # y_rand = self.load_im_size[1]-self.sample_patch_size[1]
                # x_rand = self.load_im_size[0]-self.sample_patch_size[0]
                # y_rand = 0
                # x_rand = 0
                # y_rand = safe_padding
                # x_rand = self.load_im_size[0]-self.sample_patch_size[0]

        else:
            y_rand = np.random.randint(
                0, self.load_im_size[1] - self.sample_patch_size[1]
            )
            x_rand = np.random.randint(
                0, self.load_im_size[0] - self.sample_patch_size[0]
            )

            for lm in landmarks:
                landmark_in = 0
                if (
                    y_rand + lm_safe_region
                    <= lm[1]
                    <= y_rand + self.sample_patch_size[1] - lm_safe_region
                ):
                    if (
                        x_rand + lm_safe_region
                        <= lm[0]
                        <= (x_rand + self.sample_patch_size[0]) - lm_safe_region
                    ):
                        landmark_in = 1
                landmarks_in_indicator.append(landmark_in)

        # Add the safe padding size
        y_rand_safe = y_rand + safe_padding
        x_rand_safe = x_rand + safe_padding

        # First pad image
        padded_image = np.expand_dims(
            np.pad(image[0], (safe_padding, safe_padding)), axis=0
        )
        padded_patch_size = [x + (2 * safe_padding) for x in self.sample_patch_size]

        # We pad before and after the slice.
        y_rand_pad = y_rand_safe - safe_padding
        x_rand_pad = x_rand_safe - safe_padding
        cropped_padded_sample = padded_image[
            :,
            y_rand_pad: y_rand_pad + padded_patch_size[1],
            x_rand_pad: x_rand_pad + padded_patch_size[0],
        ]

        # Calculate the new origin: 2*safe_padding bc we padded image & then added pad to the patch.
        normalized_landmarks = [
            [
                (lm[0] + 2 * safe_padding) - (x_rand_safe),
                (lm[1] + 2 * safe_padding) - (y_rand_safe),
            ]
            for lm in landmarks
        ]

        if self.debug:
            padded_lm = [
                [lm[0] + safe_padding, lm[1] + safe_padding] for lm in landmarks
            ]

            self.logger.info(
                "\n \n \n the min xy is [%s,%s]. padded is [%s, %s] normal landmark is %s, padded lm is %s \
                and the normalized landmark is %s : ",                    y_rand_safe,
                x_rand_safe,
                x_rand_pad,
                y_rand_pad,
                landmarks,
                padded_lm,
                normalized_landmarks,
            )

            visualize_patch(
                image[0],
                landmarks[0],
                padded_image[0],
                padded_lm[0],
                cropped_padded_sample[0],
                normalized_landmarks[0],
                [x_rand_pad, y_rand_pad],
            )
        return (
            cropped_padded_sample,
            normalized_landmarks,
            landmarks_in_indicator,
            [x_rand, y_rand],
        )

    def get_patch_stitching_info(self, image_size, patch_size):
        """
        Get stitching info for breaking up an input image into patches,
        where each patch overlaps with the next by 50% in x,y. The x,y, indicies are returned for each
        patch so we know how to slice the full resolution image.

        Args:
            image (_type_): _description_
        Returns
            patch_start_idxs ([[x,y]]) list of x,y indicies of where to slice for each patch

        """
        patch_start_idxs = []

        break_x = break_y = False
        for x in range(0, int(image_size[0]), int(patch_size[0] / 2)):
            for y in range(0, int(image_size[1]), int(patch_size[1] / 2)):
                break_y = False
                # Ensure we do not go past the boundaries of the image
                if x > image_size[0] - patch_size[0]:
                    x = image_size[0] - patch_size[0]
                    break_x = True
                if y > image_size[1] - patch_size[1]:
                    y = image_size[1] - patch_size[1]
                    break_y = True
                # torch loads images y-x so swap axis here
                patch_start_idxs.append([x, y])

                if break_y:
                    break
            if break_x:
                break

        return patch_start_idxs
