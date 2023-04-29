import os
from pathlib import Path
from utils.im_utils.patch_helpers import sample_patch_with_bias, sample_patch_centred, get_patch_stitching_info

import numpy as np
import torch
import torchio as tio
from imgaug.augmentables import Keypoint, KeypointsOnImage
from torch.utils import data
from torchvision import transforms
from transforms.transformations import (
    HeatmapsToTensor,
    normalize_cmr,
)

from tqdm import tqdm
from transforms.dataloader_transforms import get_aug_package_loader

from utils.data.load_data import get_datatype_load, load_aspire_datalist, load_and_resize_image, maybe_get_coordinates_from_xlsx, resize_coordinates
from utils.im_utils.visualisation import visualize_patch


import logging
from abc import ABC, ABCMeta
from alive_progress import alive_bar


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
        sigmas,
        patch_sampler_args,
        dataset_args,
        data_aug_args,
        label_generator_args,
        LabelGenerator,
        transform_heatmaps,
        sample_mode: str,
        split: str = "training",
        cache_data: bool = False,
        debug: bool = False,
        input_size=[512, 512],
        num_res_supervisions: int = 5,
        additional_sample_attribute_keys=None,
    ):
        """Initialize the dataset. This is the base class for all datasets.

        Args:
            sigmas (_type_): _description_
            patch_sampler_args (Dict): A dict of arguments for the patch sampler.
            dataset_args (Dict): A dict of arguments for the generic dataset arguments.
            data_aug_args (Dict): A dict of arguments for the data augmentation.
            label_generator_args (Dict): A dict of arguments for the Label Generator.
            LabelGenerator (_type_): _description_
            transform_heatmaps (bool): Defaults to False. 
            sample_mode (str): _description_
            split (str, optional): _description_. Defaults to "training".
            cache_data (bool, optional): _description_. Defaults to False.
            debug (bool, optional): _description_. Defaults to False.
            input_size (list, optional): _description_. Defaults to [512, 512].
            num_res_supervisions (int, optional): _description_. Defaults to 5.
            additional_sample_attribute_keys (list, optional): _description_. Defaults to [].

        Raises:
            ValueError: _description_
            ValueError: _description_
        """

        super(DatasetBase, self).__init__()

        # Logger
        self.logger = logging.getLogger()

        # We are passing in the label generator here, this is unique to each model_trainer class.
        self.LabelGenerator = LabelGenerator
        self.transform_heatmaps = transform_heatmaps
        self.hm_lambda_scale = label_generator_args["hm_lambda_scale"]
        self.generate_hms_here = label_generator_args["generate_heatmaps_here"]

        self.data_augmentation_strategy = data_aug_args["data_augmentation_strategy"]
        self.data_augmentation_package = data_aug_args["data_augmentation_package"]

        self.sample_patch_size = patch_sampler_args["generic"]["sample_patch_size"]
        self.sample_patch_bias = patch_sampler_args["biased"]["sampling_bias"]
        self.sample_patch_from_resolution = patch_sampler_args[
            "generic"]["sample_patch_from_resolution"]
        self.center_patch_on_coords_path = patch_sampler_args["centred"]["xlsx_path"]
        self.center_patch_sheet = patch_sampler_args["centred"]["xlsx_sheet"]
        self.center_patch_jitter = patch_sampler_args["centred"]["center_patch_jitter"]

        self.root_path = Path(dataset_args["root_path"])
        self.annotation_path = Path(dataset_args["annotation_path"])
        self.image_modality = dataset_args["image_modality"]
        self.landmarks = dataset_args["landmarks"]
        self.cv = dataset_args["fold"]
        self.dataset_split_size = dataset_args["dataset_split_size"]

        # Additional sample attributes found in the json datalist to return with each sample
        self.additional_sample_attribute_keys = additional_sample_attribute_keys if additional_sample_attribute_keys is not None else []
        self.additional_sample_attributes = {
            k: [] for k in self.additional_sample_attribute_keys
        }

        ############# Set Sample Mode #############

        self.sample_mode = sample_mode

        self.split = split
        self.sigmas = sigmas

        self.cache_data = cache_data
        self.debug = debug

        self.num_res_supervisions = num_res_supervisions

        # Lists to save the image paths (or images if caching), target coordinates (scaled to input size), and full resolution coords.
        self.images = []
        self.target_coordinates = []
        # full_res will be same as target if input and original image same size
        self.full_res_coordinates = []
        self.image_paths = []
        self.uids = []
        self.annotation_available = []
        self.original_image_sizes = []
        self.image_resizing_factors = []

        if self.sample_mode == "patch_bias" or self.sample_mode == "patch_centred":
            # Get the patches origin information. Use this for stitching together in valid/testing
            self.load_im_size = self.sample_patch_from_resolution
            self.input_size = self.sample_patch_size

            # if we're sampling patches w/o aug we still need to center crop so change the aug strategy to at least do this.
            if self.data_augmentation_strategy == None:
                self.data_augmentation_package = "imgaug"
                self.data_augmentation_strategy = "CenterCropOnly"

        elif self.sample_mode == "full":
            #  If not sample_patches then just 1 big patch!
            self.load_im_size = input_size
            self.input_size = input_size
        else:
            raise ValueError("sample mode %s not recognized." %
                             self.sample_mode)

        self.heatmap_label_size = self.input_size

        ############# Set Data Augmentation #############

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
                self.data_augmentation_package, self.data_augmentation_strategy, split
            )

        self.heatmaps_to_tensor = transforms.Compose([HeatmapsToTensor()])

        # We are using cross-validation, following our convention of naming each json train with the append "foldX" where (X= self.cv)
        if self.cv >= 0:
            label_std = os.path.join("fold" + str(self.cv) + ".json")
            self.logger.info(
                "Loading %s data for CV %s ", self.split, os.path.join(
                    self.annotation_path, label_std)
            )
            datalist = load_aspire_datalist(
                os.path.join(self.annotation_path, label_std),
                data_list_key=self.split,
                base_dir=self.root_path,
            )

        # Not using CV, load the specified json file
        else:
            self.logger.info(
                "Loading %s data (no CV) for %s ", self.split, self.annotation_path
            )
            datalist = load_aspire_datalist(
                self.annotation_path, data_list_key=self.split, base_dir=self.root_path
            )

            self.logger.info("done")

        # datalist = datalist[:20]
        if self.dataset_split_size != -1:
            datalist = datalist[: self.dataset_split_size]
            self.logger.info(
                "datalist truncated to length: %s, giving: %s", self.dataset_split_size, datalist)

        # based on first image extenstion, get the load function.
        self.datatype_load = get_datatype_load(datalist[0]["image"])

        self.load_function = lambda img: img

        # bar_logger

        with alive_bar(len(datalist), force_tty=True) as loading_bar:
            loading_bar.text('Loading Data...')
            for idx, data in enumerate(datalist):
                # self.logger.info("idx: %s", idx)
                # Add coordinate labels as sample attribute, if annotations available
                if (not isinstance(data["coordinates"], list)) or (
                    "has_annotation" in data.keys(
                    ) and data["has_annotation"] == False
                ):
                    # Case when data has no annotation, i.e. inference only, just set target coords to 0,0 and annotation_available to False
                    interested_landmarks = np.array(
                        [[0, 0]] * len(self.landmarks))
                    self.annotation_available.append(False)
                    self.full_res_coordinates.append(interested_landmarks)

                    if self.split == "training" or self.split == "validation":
                        raise ValueError(
                            "Training/Validation data must have annotations. Check your data. Sample that failed: ",
                            data,
                        )
                else:
                    # Case when we have annotations.
                    interested_landmarks = np.array(data["coordinates"])[
                        self.landmarks, :2]
                    self.full_res_coordinates.append(
                        np.array(data["coordinates"])[self.landmarks, :2])
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

                loading_bar()  # pylint: disable=not-callable

        # Maybe get external coordinates from xlsx for patch_centred sampling.
        self.patch_centring_coords = maybe_get_coordinates_from_xlsx(
            self.center_patch_on_coords_path, self.uids, self.landmarks, sheet_name=self.center_patch_sheet)  # may return none

        if self.cache_data:
            self.logger.info(
                "Cached all %s data in memory. Length of %s", self.split,  len(
                    self.images)
            )
        else:
            self.logger.info(
                "Not caching %s image data in memory, will load on the fly. Length of %s", self.split, len(
                    self.images)
            )

        self.check_uids_unique()

    def __len__(self):
        return len(self.images)

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

        # Do data augmentation (always minimum of centre crop if patch sampling).
        if self.data_augmentation_strategy is not None:

            # By default, the origin is 0,0 unless we sample from the middle of the image somewhere.
            # If sampling patches we first sample the patch with a little wiggle room, & normalize the lms. The transform center-crops it back.
            if self.sample_mode == "patch_bias":
                (
                    untransformed_im,
                    untransformed_coords,
                    landmarks_in_indicator,
                    x_y_corner,
                ) = sample_patch_with_bias(untransformed_im, untransformed_coords, self.sample_patch_bias, self.load_im_size,  self.sample_patch_size, self.logger, self.debug)

            # Sample a patch centred on given coordinates for this sample.
            elif self.sample_mode == "patch_centred":
                coords_to_centre_around = resize_coordinates(
                    self.patch_centring_coords[this_uid],  resized_factor[0])
                (
                    untransformed_im,
                    untransformed_coords,
                    landmarks_in_indicator,
                    x_y_corner,
                ) = sample_patch_centred(untransformed_im, coords_to_centre_around, self.load_im_size, self.sample_patch_size, self.center_patch_jitter, self.debug, groundtruth_lms=untransformed_coords,
                                         deterministic=self.center_patch_deterministic, garuntee_gt_in=self.guarantee_landmarks_in_image, safe_padding=self.center_safe_padding)

            kps = KeypointsOnImage(
                [Keypoint(x=coo[0], y=coo[1]) for coo in untransformed_coords],
                shape=untransformed_im[0].shape,
            )

            # list where [0] is image and [1] are coords.
            transformed_sample = self.transform(
                image=untransformed_im[0], keypoints=kps)
            # print(transformed_sample[1].shape)

            # TODO: try and not renormalize if we're patch sampling, maybe?
            if self.sample_mode != "patch_bias":
                input_image = normalize_cmr(
                    transformed_sample[0], to_tensor=True)
            else:
                input_image = torch.from_numpy(np.expand_dims(
                    transformed_sample[0], axis=0)).float()

            input_coords = np.array([[coo.x, coo.y]
                                    for coo in transformed_sample[1]])

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

        # Don't do data augmentation.
        else:
            input_coords = coords
            input_image = torch.from_numpy(image).float()
            landmarks_in_indicator = [1 for xy in input_coords]

        if self.generate_hms_here:

            if self.transform_heatmaps:
                assert self.num_res_supervisions == 1, "Must use num_res_supervisions equal to 1 to use transformed heat maps"

                label = self.LabelGenerator.generate_labels(
                    untransformed_coords,
                    x_y_corner,
                    landmarks_in_indicator,
                    self.heatmap_label_size,
                    hm_sigmas,
                    self.num_res_supervisions,
                    self.hm_lambda_scale,
                )

                heatmaps = label["heatmaps"][0]

                # Convert heatmaps to numpy arrays if they are torch tensors
                heatmaps_np = [hm.numpy() if isinstance(
                    hm, torch.Tensor) else hm for hm in heatmaps]

                # Add an extra dimension to the numpy arrays to represent num_channels (required by imgaug)
                heatmaps_expanded = [np.expand_dims(
                    hm, axis=-1) for hm in heatmaps_np]

                # Apply the imgaug transformation to the batch of heatmaps
                heatmaps_augmented_batch = self.transform(
                    images=heatmaps_expanded)

                # Remove the extra dimension and convert the heatmaps back to tensors
                augmented_heatmaps = [torch.from_numpy(np.squeeze(
                    hm, axis=-1)).float() for hm in heatmaps_augmented_batch]

                # Stack the augmented heatmaps back into a single tensor
                augmented_heatmaps_tensor = torch.stack(
                    augmented_heatmaps, dim=0)

                # Update the 'heatmaps' key in the label dictionary
                label["heatmaps"] = [augmented_heatmaps_tensor]

            else:
                label = self.LabelGenerator.generate_labels(
                    input_coords,
                    x_y_corner,
                    landmarks_in_indicator,
                    self.heatmap_label_size,
                    hm_sigmas,
                    self.num_res_supervisions,
                    self.hm_lambda_scale,
                )
                # print("label: ", label)
                # print("label shape: ", label["heatmaps"][0].shape)
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
