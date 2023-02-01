import copy
import logging
import numpy as np
import math
import matplotlib.pyplot as plt
from utils.im_utils.heatmap_manipulation import get_coords
from utils.im_utils.visualisation import (
    visualize_image_trans_coords,
    visualize_imageNcoords_cropped_imgNnormcoords,
)
import torch
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import matplotlib.patches as patchesplt
import torch.nn.functional as F
import cv2
from scipy.stats import multivariate_normal


class LabelGenerator(ABC):
    """Super class that defines some methods for generating landmark labels."""

    def __init__(self, full_res_size, network_input_size):
        self.full_res_size = full_res_size
        self.network_input_size = network_input_size

    @abstractmethod
    def generate_labels(
        self,
        landmarks,
        x_y_corner_patch,
        landmarks_in_indicator,
        input_size,
        hm_sigmas,
        num_res_supervisions,
        hm_lambda_scale,
    ):
        """generates heatmaps for given landmarks of size input_size, using sigma hm_sigmas.
            Generates int(num_res_supervisions) heatmaps, each half the size as previous.
            The hms are scaled by float hm_lambda_scale

        Args:
            landmarks [[int, int]]: list of landmarks to gen heatmaps for
            input_size [int, int]: size of first heatmap
            hm_sigmas [float]: gaussian sigmas of heatmaps, 1 for each landmark
            num_res_supervisions int: number of heatmaps to generate, each half resolution of previous.
            hm_lambda_scale float: value to scale heatmaps by.


            landmarks ([[int,int]]): A 2D list of ints where each entry is the [x,y] coordinate of a landmark.
            x_y_corner_patch ([int, int]): The coordinates of the top left of the image sample you are creating a heatmap for.
            landmarks_in_indicator ([int]): A list of 1s and 0s where 1 indicates the landmark was in the model input image and 0 if not.
            image_size ([int, int]): Size of the heatmap to produce.
            sigmas ([float]): List of sigmas for the size of the heatmap. Each sigma is for the heatmap for a level of deep supervision.
                            The first sigma defines the sigma for the full-size resolution heatmap, the next for the half-resolution heatmap,
                            the next for the 1/8 resolution heatmap etc.
            num_res_levels (int): Number of deep supervision levels (so should be the length of the list of sigmas. Kind-of redundant).
            lambda_scale (float): Scaler to multiply the heatmap magnitudes by.
            dtype: datatype of the output label
            to_tensor (bool): Whether to output the label as a tensor object or numpy object.

        """

    @abstractmethod
    def debug_sample(self, sample_dict, image, coordinates):
        """Visually debug a sample. Provide logging and visualisation of the sample.

        Args:
            sample_dict (dict): dict of sample info returned by __get_item__ method
            landmarks [int, int]: list of the original landmarks, before any augmentation (same as those in sample_dict if no aug used).
            image [float, float]: original input image before augmentation  (same as those in sample_dict if no aug used).
        """

    @abstractmethod
    def stitch_heatmap(self, patch_predictions, stitching_info):
        """
        Use model outputs from a patchified image to stitch together a full resolution heatmap

        """

    @abstractmethod
    def debug_prediction(
        self,
        input_dict,
        prediction_output,
        predicted_coords,
        input_size_pred_coords,
        logged_vars,
        extra_info,
    ):
        """
        Debug prediction from the model.

        """


class GPLabelGenerator(LabelGenerator):
    """Label generator for Gaussian process. the label is literally just the coordinates."""

    def __init__(self):
        super(LabelGenerator, self).__init__()
        self.logger = logging.getLogger()

    def generate_labels(
        self,
        landmarks,
        x_y_corner_patch,
        landmarks_in_indicator,
        input_size,
        hm_sigmas,
        num_res_supervisions,
        hm_lambda_scale=100,
        dtype=np.float32,
        to_tensor=True,
    ):
        """Simply returns the coordinates of the landmarks as the label."""

        if to_tensor:
            return_dict = {"landmarks": torch.from_numpy(landmarks)}
        else:
            return_dict = {"landmarks": landmarks}

        return return_dict

    def stitch_heatmap(self, patch_predictions, stitching_info):
        """
        Use model outputs from a patchified image to stitch together a full resolution heatmap.
        Irrelevent for this label generator.

        """
        pass

    def debug_sample(self, sample_dict, image, coordinates):
        """Visually debug a sample. Provide logging and visualisation of the sample."""

        # Should debug the sample here. Need to turn it back into a patch........

        self.logger.info("Sample Dict: %s", sample_dict)

    def debug_crop(
        self, original_im, cropped_im, original_lms, normalized_lms, lms_indicators
    ):
        """Visually debug a cropped sample. Provide logging and visualisation of the sample."""

        self.logger.info("before coords %s: ", original_lms)
        self.logger.info("normalized lms %s: ", normalized_lms)
        self.logger.info("landmark indicators %s", lms_indicators)

        # visualize_image_trans_target(np.squeeze(image), sample["image"][0], heatmaps[-1])
        visualize_imageNcoords_cropped_imgNnormcoords(
            original_im[0], cropped_im[0], original_lms, normalized_lms, lms_indicators
        )

    def debug_prediction(
        self,
        input_dict,
        prediction_output,
        predicted_coords,
        input_size_pred_coords,
        logged_vars,
        extra_info,
        min_error=0,
    ):
        """Visually debug a prediction and compare to the target. Provide logging and visualisation of the sample."""
        coordinate_label = input_dict["label"]["landmarks"]

        transformed_targ_coords = np.array(input_dict["target_coords"])
        full_res_coords = np.array(input_dict["full_res_coords"])
        transformed_input_image = input_dict["image"]

        model_output_coords = [x.cpu().detach().numpy() for x in prediction_output.mean]

        predicted_coords = [x.cpu().detach().numpy() for x in predicted_coords]
        # input_size_pred_coords = extra_info["coords_og_size"]

        for sample_idx, ind_sample in enumerate(logged_vars):
            f, ax = plt.subplots(1, 2, figsize=(8, 3))
            # extra_info = {"lower": lower, "upper": upper, "cov_matr": cov_matr}

            # extra_info = ind_sample["extra_info"]
            # create  kernel
            m1 = model_output_coords[sample_idx][0]
            s1 = extra_info["cov_matr"][sample_idx]
            k1 = multivariate_normal(mean=m1, cov=s1)

            # create a grid of (x,y) coordinates at which to evaluate the kernels
            xlim = (0, np.sqrt(len(transformed_input_image[sample_idx][0])))
            ylim = (0, np.sqrt(len(transformed_input_image[sample_idx][0])))
            xres = int(np.sqrt(len(transformed_input_image[sample_idx][0])))
            yres = int(np.sqrt(len(transformed_input_image[sample_idx][0])))

            x = np.linspace(xlim[0], xlim[1], xres)
            y = np.linspace(ylim[0], ylim[1], yres)
            xx, yy = np.meshgrid(x, y)

            # evaluate kernels at grid points
            xxyy = np.c_[xx.ravel(), yy.ravel()]
            zz = k1.pdf(xxyy)

            # reshape and plot image
            img = zz.reshape((xres, yres))
            ax[1].imshow(img)

            # show image with label
            image_label = coordinate_label[sample_idx][0]
            image_ex = transformed_input_image[sample_idx][0].cpu().detach().numpy()

            ax[0].imshow(image_ex.reshape((xres, yres)))

            rect1 = patchesplt.Rectangle(
                (int(image_label[0]), int(image_label[1])),
                3,
                3,
                linewidth=2,
                edgecolor="g",
                facecolor="none",
            )
            ax[0].add_patch(rect1)

            rect2 = patchesplt.Rectangle(
                (int(image_label[0]), int(image_label[1])),
                3,
                3,
                linewidth=2,
                edgecolor="g",
                facecolor="none",
            )
            ax[1].add_patch(rect2)
            rect3 = patchesplt.Rectangle(
                (int(m1[0]), int(m1[1])),
                3,
                3,
                linewidth=2,
                edgecolor="r",
                facecolor="none",
            )
            ax[1].add_patch(rect3)

            plt.show()
            plt.close()



class GPFlowLabelGenerator(LabelGenerator):
    """Label generator for Gaussian process. the label is literally just the coordinates."""

    def __init__(self):
        super(LabelGenerator, self).__init__()
        self.logger = logging.getLogger()

    def generate_labels(
        self,
        landmarks,
        x_y_corner_patch,
        landmarks_in_indicator,
        input_size,
        hm_sigmas,
        num_res_supervisions,
        hm_lambda_scale=100,
        dtype=np.float32,
        to_tensor=True,
    ):
        """Simply returns the coordinates of the landmarks as the label."""

        if to_tensor:
            return_dict = {"landmarks": torch.from_numpy(landmarks)}
        else:
            return_dict = {"landmarks": landmarks}

        return return_dict

    def stitch_heatmap(self, patch_predictions, stitching_info):
        """
        Use model outputs from a patchified image to stitch together a full resolution heatmap.
        Irrelevent for this label generator.

        """
        pass

    def debug_sample(self, sample_dict, image, coordinates):
        """Visually debug a sample. Provide logging and visualisation of the sample."""

        # Should debug the sample here. Need to turn it back into a patch........

        self.logger.info("Sample Dict: %s", sample_dict)

    def debug_crop(
        self, original_im, cropped_im, original_lms, normalized_lms, lms_indicators
    ):
        """Visually debug a cropped sample. Provide logging and visualisation of the sample."""

        self.logger.info("before coords %s: ", original_lms)
        self.logger.info("normalized lms %s: ", normalized_lms)
        self.logger.info("landmark indicators %s", lms_indicators)

        # visualize_image_trans_target(np.squeeze(image), sample["image"][0], heatmaps[-1])
        visualize_imageNcoords_cropped_imgNnormcoords(
            original_im[0], cropped_im[0], original_lms, normalized_lms, lms_indicators
        )

    def debug_prediction(
        self,
        input_dict,
        prediction_output,
        predicted_coords,
        input_size_pred_coords,
        logged_vars,
        extra_info,
        min_error=0,
    ):
        """Visually debug a prediction and compare to the target. Provide logging and visualisation of the sample."""
        coordinate_label = input_dict["label"]["landmarks"]

        transformed_targ_coords = np.array(input_dict["target_coords"])
        full_res_coords = np.array(input_dict["full_res_coords"])
        transformed_input_image = input_dict["image"]

        model_output_coords = [x.cpu().detach().numpy() for x in prediction_output.mean]

        predicted_coords = [x.cpu().detach().numpy() for x in predicted_coords]
        # input_size_pred_coords = extra_info["coords_og_size"]

        for sample_idx, ind_sample in enumerate(logged_vars):
            f, ax = plt.subplots(1, 2, figsize=(8, 3))
            # extra_info = {"lower": lower, "upper": upper, "cov_matr": cov_matr}

            # extra_info = ind_sample["extra_info"]
            # create  kernel
            m1 = model_output_coords[sample_idx][0]
            s1 = extra_info["cov_matr"][sample_idx]
            k1 = multivariate_normal(mean=m1, cov=s1)

            # create a grid of (x,y) coordinates at which to evaluate the kernels
            xlim = (0, np.sqrt(len(transformed_input_image[sample_idx][0])))
            ylim = (0, np.sqrt(len(transformed_input_image[sample_idx][0])))
            xres = int(np.sqrt(len(transformed_input_image[sample_idx][0])))
            yres = int(np.sqrt(len(transformed_input_image[sample_idx][0])))

            x = np.linspace(xlim[0], xlim[1], xres)
            y = np.linspace(ylim[0], ylim[1], yres)
            xx, yy = np.meshgrid(x, y)

            # evaluate kernels at grid points
            xxyy = np.c_[xx.ravel(), yy.ravel()]
            zz = k1.pdf(xxyy)

            # reshape and plot image
            img = zz.reshape((xres, yres))
            ax[1].imshow(img)

            # show image with label
            image_label = coordinate_label[sample_idx][0]
            image_ex = transformed_input_image[sample_idx][0].cpu().detach().numpy()

            ax[0].imshow(image_ex.reshape((xres, yres)))

            rect1 = patchesplt.Rectangle(
                (int(image_label[0]), int(image_label[1])),
                3,
                3,
                linewidth=2,
                edgecolor="g",
                facecolor="none",
            )
            ax[0].add_patch(rect1)

            rect2 = patchesplt.Rectangle(
                (int(image_label[0]), int(image_label[1])),
                3,
                3,
                linewidth=2,
                edgecolor="g",
                facecolor="none",
            )
            ax[1].add_patch(rect2)
            rect3 = patchesplt.Rectangle(
                (int(m1[0]), int(m1[1])),
                3,
                3,
                linewidth=2,
                edgecolor="r",
                facecolor="none",
            )
            ax[1].add_patch(rect3)

            plt.show()
            plt.close()



class UNetLabelGenerator(LabelGenerator):
    """Generates target heatmaps for the U-Net network training scheme"""

    def __init__(self):
        super(LabelGenerator, self).__init__()
        self.logger = logging.getLogger()

    def generate_labels(
        self,
        landmarks,
        x_y_corner_patch,
        landmarks_in_indicator,
        input_size,
        hm_sigmas,
        num_res_supervisions,
        hm_lambda_scale=100,
        dtype=np.float32,
        to_tensor=True,
    ):
        """Generates Gaussian heatmaps for given landmarks of size input_size, using sigma hm_sigmas."""

        return_dict = {"heatmaps": []}

        heatmap_list = []
        resizing_factors = [[2**x, 2**x] for x in range(num_res_supervisions)]

        # Generates a heatmap for multiple resolutions based on # down steps in encoder (1x, 0.5x, 0.25x etc)
        for size_f in resizing_factors:
            intermediate_heatmaps = []
            # Generate a heatmap for each landmark
            for idx, lm in enumerate(landmarks):

                lm = np.round(lm / size_f)
                downsample_size = [input_size[0] / size_f[0], input_size[1] / size_f[1]]
                down_sigma = hm_sigmas[idx] / size_f[0]

                # If the landmark is present in image, generate a heatmap, otherwise generate a blank heatmap.
                if landmarks_in_indicator[idx] == 1:
                    intermediate_heatmaps.append(
                        gaussian_gen(
                            lm, downsample_size, 1, down_sigma, dtype, hm_lambda_scale
                        )
                    )
                else:
                    intermediate_heatmaps.append(
                        np.zeros((int(downsample_size[0]), int(downsample_size[1])))
                    )
            heatmap_list.append(np.array(intermediate_heatmaps))

        hm_list = heatmap_list[::-1]

        if to_tensor:
            all_seg_labels = []
            for maps in hm_list:
                all_seg_labels.append(torch.from_numpy(maps).float())

            hm_list = all_seg_labels

        return_dict["heatmaps"] = hm_list
        return return_dict

    def stitch_heatmap(self, patch_predictions, stitching_info):
        """
        Use model outputs from a patchified image to stitch together a full resolution heatmap

        """

    def debug_sample(self, sample_dict, untrans_image, untrans_coords):
        """Visually debug a sample. Provide logging and visualisation of the sample."""

        # print("before coords: ", landmarks)
        # print("og image sahpe: ", image.shape, "trans image shape", sample_dict["image"].shape, "trans targ coords: ", sample_dict["target_coords"])
        # print("len of hetamps ", len(sample_dict["label"]), " and shape: ", sample_dict["label"][-1].shape, " and hm exp shape ", np.expand_dims(sample_dict["label"][-1], axis=0).shape)
        # landmarks_from_label = get_coords(torch.from_numpy(np.expand_dims(sample_dict["label"][-1], axis=0)))
        # print("landmarks reverse engineered from heatmap label: ", landmarks_from_label)

        # visualize_image_trans_target(np.squeeze(image), sample["image"][0], heatmaps[-1])
        visualize_image_trans_coords(
            untrans_image[0],
            untrans_coords,
            sample_dict["image"][0],
            sample_dict["target_coords"],
        )

    def debug_crop(
        self, original_im, cropped_im, original_lms, normalized_lms, lms_indicators
    ):
        """Visually debug a cropped sample. Provide logging and visualisation of the sample."""

        self.logger.info("before coords: %s ", original_lms)
        self.logger.info("normalized lms:  %s", normalized_lms)
        self.logger.info("landmark indicators  %s", lms_indicators)

        # visualize_image_trans_target(np.squeeze(image), sample["image"][0], heatmaps[-1])
        visualize_imageNcoords_cropped_imgNnormcoords(
            original_im[0], cropped_im[0], original_lms, normalized_lms, lms_indicators
        )

    def debug_prediction(
        self,
        input_dict,
        prediction_output,
        predicted_coords,
        input_size_pred_coords,
        logged_vars,
        extra_info,
    ):
        """Visually debug a prediction and compare to the target. Provide logging and visualisation of the sample."""
        heatmap_label = input_dict["label"]["heatmaps"][
            -1
        ]  # -1 to get the last layer only (ignore deep supervision labels)

        transformed_targ_coords = np.array(input_dict["target_coords"])
        full_res_coords = np.array(input_dict["full_res_coords"])
        transformed_input_image = input_dict["image"]

        predicted_heatmap = [x.cpu().detach().numpy() for x in prediction_output][
            -1
        ]  # -1 to get the last layer only (ignore deep supervision predictions)

        predicted_coords = [x.cpu().detach().numpy() for x in predicted_coords]
        input_size_pred_coords = extra_info["coords_og_size"]

        for sample_idx, ind_sample in enumerate(logged_vars):
            self.logger.info(
                "\n uid: %s. Mean Error: %s ", ind_sample["uid"], ind_sample["Error All Mean"])
            colours = np.arange(len(predicted_coords[sample_idx]))

            # Only show debug if any landmark error is >10 pixels!
            if (
                len(
                    [
                        x
                        for x in range(len(predicted_coords[sample_idx]))
                        if (
                            ind_sample["L" + str(x)] != None
                            and ind_sample["L" + str(x)] > 10
                        )
                    ]
                )
                > 0
            ):
                fig, ax = plt.subplots(1, ncols=1, squeeze=False)

                for coord_idx, pred_coord in enumerate(predicted_coords[sample_idx]):
                    self.logger.info(
                        "L%s: Full Res Prediction: %s, Full Res Target: %s, Error: %s. Input Res targ %s, input res pred %s."
                        % (
                            coord_idx,
                            pred_coord,
                            full_res_coords[sample_idx][coord_idx],
                            ind_sample["L" + str(coord_idx)],
                            transformed_targ_coords[sample_idx][coord_idx],
                            input_size_pred_coords[sample_idx][coord_idx],
                        )
                    )

                    # difference between these is removing the padding (so -128, or whatever the patch padding was)
                    self.logger.info(
                        "predicted (red) %s vs  target coords (green) %s ",
                        input_size_pred_coords[sample_idx][coord_idx],
                        transformed_targ_coords[sample_idx][coord_idx],
                    )

                    # 1) show input image with target lm and predicted lm
                    # 2) show predicted heatmap
                    # 3 show target heatmap

                    # 1)
                    ax[0, 0].imshow(transformed_input_image[sample_idx][0])
                    rect1 = patchesplt.Rectangle(
                        (
                            transformed_targ_coords[sample_idx][coord_idx][0],
                            transformed_targ_coords[sample_idx][coord_idx][1],
                        ),
                        6,
                        6,
                        linewidth=2,
                        edgecolor="g",
                        facecolor="none",
                    )
                    ax[0, 0].add_patch(rect1)
                    rect2 = patchesplt.Rectangle(
                        (
                            input_size_pred_coords[sample_idx][coord_idx][0]
                            .detach()
                            .numpy(),
                            input_size_pred_coords[sample_idx][coord_idx][1]
                            .detach()
                            .cpu()
                            .numpy(),
                        ),
                        6,
                        6,
                        linewidth=2,
                        edgecolor="pink",
                        facecolor="none",
                    )
                    ax[0, 0].add_patch(rect2)

                    ax[0, 0].text(
                        transformed_targ_coords[sample_idx][coord_idx][0],
                        transformed_targ_coords[sample_idx][coord_idx][1]
                        + 10,  # Position
                        "L" + str(coord_idx),  # Text
                        verticalalignment="bottom",  # Centered bottom with line
                        horizontalalignment="center",  # Centered with horizontal line
                        fontsize=12,  # Font size
                        color="g",  # Color
                    )
                    if ind_sample["L" + str(coord_idx)] > 10:
                        pred_text = "r"
                    else:
                        pred_text = "pink"
                    ax[0, 0].text(
                        input_size_pred_coords[sample_idx][coord_idx][0]
                        .detach()
                        .cpu()
                        .numpy(),
                        input_size_pred_coords[sample_idx][coord_idx][1]
                        .detach()
                        .cpu()
                        .numpy()
                        + 10,  # Position
                        "L"
                        + str(coord_idx)
                        + " E="
                        + str(np.round(ind_sample["L" + str(coord_idx)], 2)),  # Text
                        verticalalignment="bottom",  # Centered bottom with line
                        horizontalalignment="center",  # Centered with horizontal line
                        fontsize=12,  # Font size
                        color=pred_text,  # Color
                    )
                    ax[0, 0].set_title(
                        "uid: %s. Mean Error: %s +/- %s"
                        % (
                            ind_sample["uid"],
                            np.round(ind_sample["Error All Mean"], 2),
                            np.round(ind_sample["Error All Std"]),
                        )
                    )

                    # # #2)
                    # print("len of predicted heatmaps ", len(predicted_heatmap), " size of sample idx heatmaps ", predicted_heatmap[sample_idx].shape)
                    # ax[0,1].imshow(predicted_heatmap[sample_idx][coord_idx])

                    # rect45 = patchesplt.Rectangle(( predicted_coords[sample_idx][coord_idx][0], predicted_coords[sample_idx][coord_idx][1]) ,6,6,linewidth=2,edgecolor='pink',facecolor='none')
                    # ax[0,1].add_patch(rect45)
                    # rect46 = patchesplt.Rectangle((  full_res_coords[sample_idx][coord_idx][0], full_res_coords[sample_idx][coord_idx][1]) ,6,6,linewidth=2,edgecolor='g',facecolor='none')
                    # ax[0,1].add_patch(rect46)

                    # 3)
                    # ax[coord_idx,2].imshow(heatmap_label[sample_idx][coord_idx])
                    # rect35 = patchesplt.Rectangle((  full_res_coords[sample_idx][coord_idx][0], full_res_coords[sample_idx][coord_idx][1]) ,6,6,linewidth=2,edgecolor='g',facecolor='none')
                    # ax[coord_idx,2].add_patch(rect35)

                    # rect15 = patchesplt.Rectangle((  full_res_coords[sample_idx][coord_idx][0], full_res_coords[sample_idx][coord_idx][1]) ,6,6,linewidth=2,edgecolor='g',facecolor='none')
                    # ax[0,0].add_patch(rect15)

                plt.show()
                plt.close()


class PHDNetLabelGenerator(LabelGenerator):
    """Generates target heatmaps and displacements for the PHD-Net network training scheme"""

    def __init__(
        self,
        maxpool_factor,
        full_heatmap_resolution,
        class_label_scheme,
        sample_grid_size,
        log_transform_displacements_bool,
        clamp_dist,
    ):
        super(LabelGenerator, self).__init__()
        # self.sampling_bias = sampling_bias
        self.maxpool_factor = maxpool_factor
        self.full_heatmap_resolution = full_heatmap_resolution
        self.class_label_scheme = class_label_scheme
        self.sample_grid_size = sample_grid_size
        self.log_transform_displacements_bool = log_transform_displacements_bool
        self.clamp_dist = clamp_dist

    def stitch_heatmap(self, patch_predictions, stitching_info):
        """
        Use model outputs from a patchified image to stitch together a full resolution heatmap

        """

    def generate_labels(
        self,
        landmarks,
        x_y_corner_patch,
        landmarks_in_indicator,
        input_size,
        hm_sigmas,
        num_res_supervisions,
        hm_lambda_scale=100,
        dtype=np.float32,
        to_tensor=True,
    ):
        """Generates heatmap labels i.e. target heatmaps and displacements for each patch.

        Args:
             landmarks ([[int,int]]): A 2D list of ints where each entry is the [x,y] coordinate of a landmark.
             x_y_corner_patch ([int, int]): The coordinates of the top left of the image sample you are creating a heatmap for.
             landmarks_in_indicator ([int]): A list of 1s and 0s where 1 indicates the landmark was in the model input image and 0 if not.
             image_size ([int, int]): Size of the heatmap to produce.
             sigmas ([float]): List of sigmas for the size of the heatmap. Each sigma is for the heatmap for a level of deep supervision.
                             The first sigma defines the sigma for the full-size resolution heatmap, the next for the half-resolution heatmap,
                             the next for the 1/8 resolution heatmap etc.
             num_res_levels (int): Number of deep supervision levels (so should be the length of the list of sigmas. Kind-of redundant).
             lambda_scale (float): Scaler to multiply the heatmap magnitudes by.
             dtype: datatype of the output label
             to_tensor (bool): Whether to output the label as a tensor object or numpy object.
         Returns:
             return_dict (Dict): A dictionary containing the following keys and values:
                 {
                     "patch_heatmap" ([]): For each landmark, a 2D list of all the patches, each with one float for the heatmap value of that patch.
                     "patch_displacements": For each landmark, a 2D list of all the patches, each with an entry of 2 floats for the x,y displacements from patch to lm.
                     "patch_displacements":  For each landmark, a 2D list of all the patches, each with one float for the weighting to apply to patch_displacements.
                 }

        """

        return_dict = {
            "patch_heatmap": [],
            "patch_displacements": [],
            "displacement_weights": [],
        }

        for idx, lm in enumerate(landmarks):
            sigma = hm_sigmas[idx]

            x_y_displacements, sub_class, weights = gen_patch_displacements_heatmap(
                lm,
                x_y_corner_patch,
                self.class_label_scheme,
                self.sample_grid_size,
                self.full_heatmap_resolution,
                self.maxpool_factor,
                sigma,
                lambda_scale=hm_lambda_scale,
                log_transform_displacements_bool=self.log_transform_displacements_bool,
                clamp_dist=self.clamp_dist,
                debug=False,
            )

            return_dict["patch_heatmap"].append(sub_class)
            return_dict["patch_displacements"].append(x_y_displacements)
            return_dict["displacement_weights"].append(weights)
        return_dict["xy_corner"] = x_y_corner_patch

        if to_tensor:
            return_dict["patch_heatmap"] = torch.stack(
                [torch.from_numpy(x).float() for x in return_dict["patch_heatmap"]]
            )
            return_dict["patch_displacements"] = torch.stack(
                [
                    torch.from_numpy(x).float()
                    for x in return_dict["patch_displacements"]
                ]
            )
            return_dict["displacement_weights"] = torch.stack(
                [
                    torch.from_numpy(x).float()
                    for x in return_dict["displacement_weights"]
                ]
            )

        # shape is 1,1,16,16
        return return_dict

    def debug_sample(self, sample_dict, image, coordinates):
        """Visually debug a sample. Provide logging and visualisation of the sample."""

        xy_corner = sample_dict["label"]["xy_corner"]
        patch_heatmap_label = sample_dict["label"]["patch_heatmap"]
        patch_disp_label = sample_dict["label"]["patch_displacements"][0]
        patch_disp_weights = sample_dict["label"]["displacement_weights"][0]
        transformed_targ_coords = np.array(sample_dict["target_coords"][0])
        full_res_coords = np.array(sample_dict["full_res_coords"][0])
        transformed_input_image = sample_dict["image"][0]
        untransformed_im = image[0]
        untransformed_coords = np.array(coordinates[0])

        print(
            "all shapes: ",
            untransformed_im.shape,
            untransformed_coords.shape,
            patch_heatmap_label.shape,
            patch_disp_label.shape,
            patch_disp_weights.shape,
            transformed_targ_coords.shape,
            full_res_coords.shape,
            transformed_input_image.shape,
        )

        # difference between these is removing the padding (so -128, or whatever the patch padding was)
        print(
            "untransformed and transformed coords: ",
            untransformed_coords,
            transformed_targ_coords,
        )
        print("xy corner: ", xy_corner)
        # 1) reconstructed lm on untrans image
        # 2) show untrans image with untrans lm
        # 3) show input image (transformed_input_image) with target coords (transformed_targ_coords)
        # 4) show patch-wise heatmap label matches landmarks (patch_heatmap_label with downsampled transformed_targ_coords)
        # 5) show patch-wise heatmap label matches landmarks upscaled show patch_heatmap_label interpolated up to transformed_targ_coords)
        # 6) show patch_disp_weights point to transformed_targ_coords

        fig, ax = plt.subplots(nrows=2, ncols=3)

        # 2)
        print("Full resolution coords: ", full_res_coords)
        ax[0, 0].imshow(untransformed_im)
        rect1 = patchesplt.Rectangle(
            (untransformed_coords[0], untransformed_coords[1]),
            6,
            6,
            linewidth=2,
            edgecolor="m",
            facecolor="none",
        )
        ax[0, 0].add_patch(rect1)

        # 3)
        ax[0, 1].imshow(transformed_input_image)
        rect2 = patchesplt.Rectangle(
            (transformed_targ_coords[0], transformed_targ_coords[1]),
            6,
            6,
            linewidth=2,
            edgecolor="m",
            facecolor="none",
        )
        ax[0, 1].add_patch(rect2)

        # 4)
        downsampled_coords = transformed_targ_coords / (2**self.maxpool_factor)
        print("downscampled coords to fit on heatmap label: ", downsampled_coords)
        ax[0, 2].imshow(patch_heatmap_label[0])
        rect3 = patchesplt.Rectangle(
            (downsampled_coords[0], downsampled_coords[1]),
            1,
            1,
            linewidth=2,
            edgecolor="m",
            facecolor="none",
        )
        ax[0, 2].add_patch(rect3)

        # 5)
        # tensor_weights = torch.tensor(np.expand_dims(np.expand_dims(patch_heatmap_label, axis=0), axis=0))
        # need to flip axis here because torch does y-x not x-y
        # upscaled_hm =  (F.interpolate(tensor_weights, [1,self.sample_grid_size[0], self.sample_grid_size[1]], mode="nearest-exact")).cpu().detach().numpy()[0,0]
        # upscaled_hm =  cv2.resize(patch_heatmap_label,[self.sample_grid_size[0], self.sample_grid_size[1]],0,0, interpolation = cv2.INTER_NEAREST)
        # pil_im = Image.fromarray(patch_heatmap_label,'L')
        # plt.imshow(pil_im)
        # plt.show
        # upscaled_hm = pil_im.resize([self.sample_grid_size[0], self.sample_grid_size[1]], resample=Image.NEAREST)

        step_size = 2**self.maxpool_factor
        upscaled_hm = np.broadcast_to(
            patch_heatmap_label[0][:, None, :, None],
            (
                patch_heatmap_label[0].shape[0],
                step_size,
                patch_heatmap_label[0].shape[1],
                step_size,
            ),
        ).reshape(self.sample_grid_size)
        coords_from_uhm, arg_max = get_coords(
            torch.tensor(np.expand_dims(np.expand_dims(upscaled_hm, axis=0), axis=0))
        )
        print("get_coords from upscaled_hm: ", coords_from_uhm)

        ax[1, 0].imshow(upscaled_hm)
        rect4 = patchesplt.Rectangle(
            ((transformed_targ_coords[0]), (transformed_targ_coords[1])),
            6,
            6,
            linewidth=2,
            edgecolor="m",
            facecolor="none",
        )
        ax[1, 0].add_patch(rect4)

        # 6
        # ax[1,1].imshow(upscaled_hm)
        rect5 = patchesplt.Rectangle(
            ((transformed_targ_coords[0]), (transformed_targ_coords[1])),
            6,
            6,
            linewidth=2,
            edgecolor="m",
            facecolor="none",
        )
        ax[1, 1].add_patch(rect5)

        all_locs = []
        ax[1, 1].imshow(transformed_input_image)
        for x_idx, x in enumerate(
            range(0, self.sample_grid_size[0], (2**self.maxpool_factor))
        ):
            for y_idx, y in enumerate(
                range(0, self.sample_grid_size[1], (2**self.maxpool_factor))
            ):

                center_xy = [
                    x + ((2**self.maxpool_factor) // 2),
                    y + ((2**self.maxpool_factor) // 2),
                ]
                # REMEMBER TO ADD 1 TO REVERSE THE LOG SHIFT WHEN CALCULATING THE LABELS!

                if self.log_transform_displacements_bool:
                    x_disp = np.sign(patch_disp_label[0, x_idx, y_idx]) * (
                        2 ** (abs(patch_disp_label[0, x_idx, y_idx])) - 1
                    )
                    y_disp = np.sign(patch_disp_label[1, x_idx, y_idx]) * (
                        2 ** (abs(patch_disp_label[1, x_idx, y_idx])) - 1
                    )
                else:
                    x_disp = patch_disp_label[0, x_idx, y_idx]
                    y_disp = patch_disp_label[1, x_idx, y_idx]

                loc = [center_xy[0] + x_disp, center_xy[1] + y_disp]
                ax[1, 1].arrow(center_xy[0], center_xy[1], x_disp, y_disp)
                all_locs.append(loc)
        print("average location: ", np.mean(all_locs, axis=0))
        plt.show()
        plt.close()

    def debug_prediction(
        self,
        input_dict,
        prediction_output,
        predicted_coords,
        input_size_pred_coords,
        logged_vars,
        extra_info,
    ):
        """Visually debug a prediction and compare to the target. Provide logging and visualisation of the sample."""
        xy_corner = input_dict["label"]["xy_corner"]
        patch_heatmap_label = input_dict["label"]["patch_heatmap"]
        patch_disp_label = input_dict["label"]["patch_displacements"]
        patch_disp_weights = input_dict["label"]["displacement_weights"]
        transformed_targ_coords = np.array(input_dict["target_coords"])
        full_res_coords = np.array(input_dict["full_res_coords"])
        transformed_input_image = input_dict["image"]
        predicted_heatmap = prediction_output[0].cpu().detach().numpy()
        predicted_displacements = prediction_output[1].cpu().detach().numpy()
        predicted_coords = predicted_coords.cpu().detach().numpy()
        input_size_pred_coords = input_size_pred_coords.cpu().detach().numpy()
        candidate_smoothed_maps = (
            extra_info["debug_candidate_smoothed_maps"].cpu().detach().numpy()
        )

        print(
            "all shapes: ",
            patch_heatmap_label.shape,
            patch_disp_label.shape,
            patch_disp_weights.shape,
            input_size_pred_coords.shape,
            transformed_targ_coords.shape,
            full_res_coords.shape,
            transformed_input_image.shape,
            predicted_heatmap.shape,
            predicted_displacements.shape,
            predicted_coords.shape,
        )

        for sample_idx, ind_sample in enumerate(logged_vars):
            print(
                "\n uid: %s. Mean Error: %s "
                % (ind_sample["uid"], ind_sample["Error All Mean"])
            )
            for coord_idx, pred_coord in enumerate(predicted_coords[sample_idx]):
                print(
                    "L%s: Full Res Prediction: %s, Full Res Target: %s, Error: %s. Input Res Pred %s, input res targ %s."
                    % (
                        coord_idx,
                        pred_coord,
                        full_res_coords[sample_idx][coord_idx],
                        ind_sample["L" + str(coord_idx)],
                        transformed_targ_coords[sample_idx][coord_idx],
                        input_size_pred_coords[sample_idx][coord_idx],
                    )
                )

                # difference between these is removing the padding (so -128, or whatever the patch padding was)
                print(
                    "predicted (red) vs  target coords (green): ",
                    input_size_pred_coords[sample_idx][coord_idx],
                    transformed_targ_coords[sample_idx][coord_idx],
                )

                # 1) show input image with target lm and predicted lm
                # 2) show predicted heatmap
                # 3) show candidate smoothed map
                # 4) show predicted displacements

                fig, ax = plt.subplots(nrows=2, ncols=2)

                # 1)
                ax[0, 0].imshow(transformed_input_image[sample_idx][0])
                rect1 = patchesplt.Rectangle(
                    (
                        transformed_targ_coords[sample_idx][coord_idx][0],
                        transformed_targ_coords[sample_idx][coord_idx][1],
                    ),
                    6,
                    6,
                    linewidth=2,
                    edgecolor="g",
                    facecolor="none",
                )
                ax[0, 0].add_patch(rect1)
                rect2 = patchesplt.Rectangle(
                    (
                        input_size_pred_coords[sample_idx][coord_idx][0],
                        input_size_pred_coords[sample_idx][coord_idx][1],
                    ),
                    6,
                    6,
                    linewidth=2,
                    edgecolor="r",
                    facecolor="none",
                )
                ax[0, 0].add_patch(rect2)

                # #2)

                downsample_factor = 2**self.maxpool_factor
                downsampled_pred = (
                    predicted_coords[sample_idx][coord_idx] / downsample_factor
                )
                downsampled_targ = (
                    transformed_targ_coords[sample_idx][coord_idx] / downsample_factor
                )
                max_square, _ = get_coords(
                    torch.from_numpy(
                        np.expand_dims(
                            np.expand_dims(
                                predicted_heatmap[sample_idx][coord_idx], axis=0
                            ),
                            axis=0,
                        )
                    )
                )
                max_square = max_square.cpu().detach().numpy()[0, 0]
                print(
                    "\n Downsampled on output HM: targ (g): %s, pred (r): %s, max square (m): %s"
                    % (downsampled_targ, downsampled_pred, max_square)
                )

                ax[0, 1].imshow(predicted_heatmap[sample_idx][coord_idx])
                rect3 = patchesplt.Rectangle(
                    (downsampled_pred[0], downsampled_pred[1]),
                    6,
                    6,
                    linewidth=2,
                    edgecolor="r",
                    facecolor="none",
                )
                ax[0, 1].add_patch(rect3)
                rect4 = patchesplt.Rectangle(
                    (downsampled_targ[0], downsampled_targ[1]),
                    6,
                    6,
                    linewidth=2,
                    edgecolor="g",
                    facecolor="none",
                )
                ax[0, 1].add_patch(rect4)
                rect45 = patchesplt.Rectangle(
                    (max_square[0], max_square[1]),
                    6,
                    6,
                    linewidth=2,
                    edgecolor="m",
                    facecolor="none",
                )
                ax[0, 1].add_patch(rect45)

                # 3)
                ax[1, 0].imshow(candidate_smoothed_maps[sample_idx][coord_idx])
                rect5 = patchesplt.Rectangle(
                    (
                        predicted_coords[sample_idx][coord_idx][0],
                        predicted_coords[sample_idx][coord_idx][1],
                    ),
                    6,
                    6,
                    linewidth=2,
                    edgecolor="r",
                    facecolor="none",
                )
                ax[1, 0].add_patch(rect5)
                rect6 = patchesplt.Rectangle(
                    (
                        transformed_targ_coords[sample_idx][coord_idx][0],
                        transformed_targ_coords[sample_idx][coord_idx][1],
                    ),
                    6,
                    6,
                    linewidth=2,
                    edgecolor="g",
                    facecolor="none",
                )
                ax[1, 0].add_patch(rect6)
                # 4)
                ax[1, 1].imshow(transformed_input_image[sample_idx][0])

                for x_idx, x in enumerate(
                    range(0, self.sample_grid_size[0], (2**self.maxpool_factor))
                ):
                    for y_idx, y in enumerate(
                        range(0, self.sample_grid_size[1], (2**self.maxpool_factor))
                    ):

                        center_xy = [
                            x + ((2**self.maxpool_factor) // 2),
                            y + ((2**self.maxpool_factor) // 2),
                        ]
                        # REMEMBER TO ADD 1 TO REVERSE THE LOG SHIFT WHEN CALCULATING THE LABELS!

                        if self.log_transform_displacements_bool:
                            x_disp = np.sign(
                                predicted_displacements[sample_idx][coord_idx][
                                    0, x_idx, y_idx
                                ]
                            ) * (
                                2
                                ** (
                                    abs(
                                        predicted_displacements[sample_idx][coord_idx][
                                            0, x_idx, y_idx
                                        ]
                                    )
                                )
                                - 1
                            )
                            y_disp = np.sign(
                                predicted_displacements[sample_idx][coord_idx][
                                    1, x_idx, y_idx
                                ]
                            ) * (
                                2
                                ** (
                                    abs(
                                        predicted_displacements[sample_idx][coord_idx][
                                            1, x_idx, y_idx
                                        ]
                                    )
                                )
                                - 1
                            )
                        else:
                            x_disp = predicted_displacements[sample_idx][coord_idx][
                                0, x_idx, y_idx
                            ]
                            y_disp = predicted_displacements[sample_idx][coord_idx][
                                1, x_idx, y_idx
                            ]

                        loc = [center_xy[0] + x_disp, center_xy[1] + y_disp]
                        ax[1, 1].arrow(center_xy[0], center_xy[1], x_disp, y_disp)

                # 5)

                plt.show()
                plt.close()


def generate_heatmaps(
    landmarks, image_size, sigma, num_res_levels, lambda_scale=100, dtype=np.float32
):
    heatmap_list = []

    num_heatmaps = len(landmarks)
    resizings = [
        [(num_heatmaps, int(image_size[0] / (2**x)), int(image_size[1] / (2**x)))]
        for x in range(num_res_levels)
    ]
    resizing_factors = [[2**x, 2**x] for x in range(num_res_levels)]

    for size_f in resizing_factors:
        intermediate_heatmaps = []
        for idx, lm in enumerate(landmarks):
            lm = np.round(lm / size_f)
            downsample_size = [image_size[0] / size_f[0], image_size[1] / size_f[1]]
            down_sigma = sigma[idx] / size_f[0]
            intermediate_heatmaps.append(
                gaussian_gen(lm, downsample_size, 1, down_sigma, dtype, lambda_scale)
            )
        heatmap_list.append(np.array(intermediate_heatmaps))

    return heatmap_list[::-1]


# generate Guassian with center on landmark. sx and sy are the std.
def gaussian_gen(
    landmark, resolution, step_size, std, dtype=np.float32, lambda_scale=100
):

    sx = std
    sy = std

    x = resolution[0] / step_size
    y = resolution[1] / step_size

    mx = landmark[0] / step_size
    my = landmark[1] / step_size

    x = np.arange(x)
    y = np.arange(y)

    x, y = np.meshgrid(x, y)  # get 2D variables instead of 1D

    # define guassian
    g = (
        (1)
        / (2.0 * np.pi * sx * sy)
        * np.exp(
            -((x - mx) ** 2.0 / (2.0 * sx**2.0) + (y - my) ** 2.0 / (2.0 * sy**2.0))
        )
    )

    # normalise between 0 and 1
    g *= 1.0 / g.max() * lambda_scale

    g[g <= 0] = -1

    return g


def gaussian_gen_fast(predictions, resolution, sigma):
    sx = sigma
    sy = sigma

    x = resolution[0]
    y = resolution[1]

    # predictions = predictions.cpu().detach().numpy()
    zeros = np.zeros(resolution)

    # print("zeros shape:", zeros.shape)
    # print("incside gauss gen alt. Predictions: ", predictions)

    # gen a generic gaussian blob
    blob_size = 10
    x = np.arange(blob_size)
    y = np.arange(blob_size)
    mx = my = blob_size // 2

    x, y = np.meshgrid(x, y)  # get 2D variables instead of 1D

    # define guassian
    g = (
        (1)
        / (2.0 * np.pi * sx * sy)
        * np.exp(
            -((x - mx) ** 2.0 / (2.0 * sx**2.0) + (y - my) ** 2.0 / (2.0 * sy**2.0))
        )
    )
    # normalise between 0 and 1
    g *= 1.0 / g.max()

    # TODO: do testing for edge cases below, beyond a visual check.
    # predictions =[[0,0], [5,5], [3,3], [9,9], [6,6], [510,510], [511,511], [508, 60]]
    for pred in predictions:
        pred = [round(pred[0]), round(pred[1])]
        # make sure prediction is in bounds.
        if 0 <= pred[0] < resolution[0] and 0 <= pred[1] < resolution[1]:

            # where to slice the gaussian blob? This is if landmark is on the edge, it slices the
            # corresponding part of the blob.
            if pred[0] < mx:
                x_start, x_end = (mx - pred[0]), blob_size
            elif pred[0] > (resolution[0] - mx):
                x_start, x_end = 0, mx + (resolution[0] - pred[0])

            else:
                x_start, x_end = 0, blob_size

            if pred[1] < my:
                # print("y1")
                y_start, y_end = (my - pred[1]), blob_size
            elif pred[1] > (resolution[1] - my):
                # print("y2")
                y_start, y_end = 0, my + (resolution[1] - pred[1])
            else:
                y_start, y_end = 0, blob_size

            clipped_blob = g[y_start:y_end, x_start:x_end]

            # Only add to zeros where blob is overlapping.
            start_x = max(0, min(round(pred[0] - mx), resolution[0]))
            end_x = start_x + (x_end - x_start)

            start_y = max(0, min(round(pred[1] - my), resolution[1]))
            end_y = start_y + (y_end - y_start)

            zeros[start_y:end_y, start_x:end_x] += clipped_blob

    return zeros


def gaussian_gen_alternate(predictions, resolution, sigma):
    sx = sigma
    sy = sigma

    # predictions = predictions.cpu().detach().numpy()
    zeros = np.zeros(resolution)

    # print("zeros shape:", zeros.shape)
    # print("incside gauss gen alt. Predictions: ", predictions)

    for pred in predictions:
        # print("inner pred:", pred.shape, pred)
        zeros[
            int(np.clip(np.round(pred[0]), 0, resolution[0] - 1)),
            int(np.clip(np.round(pred[1]), 0, resolution[1] - 1)),
        ] += 1

    zeros = cv2.GaussianBlur((zeros), [0, 0], sigma)
    zeros = np.transpose(zeros)
    return zeros

    # g =  np.expand_dims(zeros, axis=0)


def gen_patch_displacements_heatmap(
    landmark,
    xy_patch_corner,
    class_loss_scheme,
    grid_size,
    full_heatmap_resolution,
    maxpooling_factor,
    sigma,
    lambda_scale=100,
    log_transform_displacements_bool=True,
    debug=False,
    clamp_dist=48,
):
    """Function to generate sub-patch displacements and patch-wise heatmap values.

    Args:
        landmark (_type_): _description_
        xy_patch_corner (_type_): _description_
        class_loss_scheme (_type_): _description_
        grid_size (_type_): _description_
        full_heatmap_resolution (_type_): _description_
        maxpooling_factor (_type_): _description_
        sigma (_type_): _description_
        lambda_scale (int, optional): _description_. Defaults to 100.
        debug (bool, optional): _description_. Defaults to False.

    Raises:
        NotImplementedError: _description_

    Returns:
        _type_: _description_
    """

    # need to find sub image grid e.g. grid size 128x128, patch size of 8x8 = 16x16 patches.
    # I need this grid so i can find center of each patch, displacment to lanmdark
    # and heatmap values centered around the landmark.
    # We use log displacements to dampen distance parts, shifting function to right to avoid asymptope.
    patches = [
        grid_size[0] // 2**maxpooling_factor,
        grid_size[1] // 2**maxpooling_factor,
    ]
    step_size = 2**maxpooling_factor

    x_y_displacements = np.zeros((2, patches[0], patches[1]), dtype=np.float32)
    for x_idx, x in enumerate(range(0, grid_size[0], step_size)):
        for y_idx, y in enumerate(range(0, grid_size[0], step_size)):

            center_xy = [x + (step_size // 2), y + (step_size // 2)]

            # find log of displacements accounting for orientation
            # distance_y = min(abs(landmark[1] - center_xy[1]) , clamp_dist)
            # distance_x = min(abs(landmark[0] - center_xy[0]), clamp_dist)

            distance_y = abs(landmark[1] - center_xy[1])
            distance_x = abs(landmark[0] - center_xy[0])
            # v = [distance_x, distance_y]
            # distance_x, distance_y = ()

            # clamp to magnitude of clamp_dist
            if clamp_dist != None:
                n = math.sqrt(distance_x**2 + distance_y**2)
                if n != 0:
                    f = min(n, clamp_dist) / n
                else:
                    f = 1
                distance_x, distance_y = [f * distance_x, f * distance_y]

            if log_transform_displacements_bool:
                # shift log function by 1 so now the asymtope is at -1 instead of 0.
                if landmark[1] > center_xy[1]:
                    displace_y = math.log(distance_y + 1, 2)
                elif landmark[1] < center_xy[1]:
                    displace_y = -math.log(distance_y + 1, 2)
                else:
                    displace_y = 0

                # shift log function by 1 so now the asymtope is at -1 instead of 0.
                if landmark[0] > center_xy[0]:
                    displace_x = math.log(distance_x + 1, 2)
                elif landmark[0] < center_xy[0]:
                    displace_x = -math.log(distance_x + 1, 2)
                else:
                    displace_x = 0

            else:
                if landmark[1] > center_xy[1]:
                    displace_y = distance_y
                elif landmark[1] < center_xy[1]:
                    displace_y = -distance_y
                else:
                    displace_y = 0

                # shift log function by 1 so now the asymtope is at -1 instead of 0.
                if landmark[0] > center_xy[0]:
                    displace_x = distance_x
                elif landmark[0] < center_xy[0]:
                    displace_x = -distance_x
                else:
                    displace_x = 0

            x_y_displacements[:, x_idx, y_idx] = [displace_x, displace_y]

    ###########Gaussian weights #############
    # Generate guassian heatmap for classificaiton and displacement weights!
    # First pad around the grid (to catch landmark if the heatmap will spill over the edge).
    # then downsample to the size of the heatmap to match how many patches we have.
    safe_padding = 128
    padded_lm = landmark + safe_padding
    downscaled_padded_lm = padded_lm / step_size
    hm_res = [
        (grid_size[0] + (safe_padding * 2)) / step_size,
        (grid_size[1] + (safe_padding * 2)) / step_size,
    ]

    gaussian_weights_full = gaussian_gen(
        downscaled_padded_lm, hm_res, 1, sigma, lambda_scale=lambda_scale
    )
    gaussian_weights = gaussian_weights_full[
        (safe_padding // step_size): (safe_padding + grid_size[0]) // step_size,
        safe_padding // step_size: (safe_padding + grid_size[1]) // step_size,
    ]
    displacement_weights = gaussian_weights / lambda_scale

    # Classification labels. Can be gaussian heatmap or binary.
    if class_loss_scheme == "gaussian":
        sub_class = gaussian_weights
    else:
        raise NotImplementedError(
            'only gaussian labels for class loss scheme are currently implemented. try with MODEL.PHDNET.CLASS_LOSS_SCHEME as "gaussian"'
        )

    # ####################### DEBUGGING VISUALISATION ##############

    if debug:
        print("normalized landmark: ", landmark)
        print("padded full landmark: ", padded_lm)
        print(
            "full gauss shape and sliced gauss shape ",
            gaussian_weights_full.shape,
            gaussian_weights.shape,
        )

        fig, ax = plt.subplots(nrows=2, ncols=2)
        ax[0, 0].imshow(gaussian_weights_full)
        ax[0, 1].imshow(gaussian_weights)
        # resized_gauss = torch.tensor(gaussian_weights).resize(grid_size)
        tensor_weights = torch.tensor(
            np.expand_dims(np.expand_dims(gaussian_weights, axis=0), axis=0)
        )
        print("weights and resize requestion: ", tensor_weights.shape, grid_size)
        resized_gauss = (
            (
                F.interpolate(
                    tensor_weights, [grid_size[0], grid_size[1]], mode="nearest"
                )
            )
            .cpu()
            .detach()
            .numpy()[0, 0]
        )
        ax[1, 0].imshow(copy.deepcopy(resized_gauss))

        ax[1, 1].imshow(resized_gauss)

        downscaled_full_lms = np.round(padded_lm / (2**maxpooling_factor))
        print("downscaled lm to fit 64x64: ", downscaled_full_lms)
        rect0 = patchesplt.Rectangle(
            (downscaled_full_lms[0], downscaled_full_lms[1]),
            6,
            6,
            linewidth=2,
            edgecolor="r",
            facecolor="none",
        )
        ax[0, 0].add_patch(rect0)

        downscaled_full_lms_16 = np.round(landmark / (2**maxpooling_factor))
        print("downscaled lm to fit 16x16: ", downscaled_full_lms_16)

        rect1 = patchesplt.Rectangle(
            (downscaled_full_lms_16[0], downscaled_full_lms_16[1]),
            6,
            6,
            linewidth=2,
            edgecolor="m",
            facecolor="none",
        )
        ax[0, 1].add_patch(rect1)

        rect2 = patchesplt.Rectangle(
            (landmark[0], landmark[1]),
            6,
            6,
            linewidth=2,
            edgecolor="m",
            facecolor="none",
        )
        ax[1, 0].add_patch(rect2)

        rect3 = patchesplt.Rectangle(
            (landmark[0], landmark[1]),
            6,
            6,
            linewidth=2,
            edgecolor="m",
            facecolor="none",
        )
        ax[1, 1].add_patch(rect3)

        for x_idx, x in enumerate(range(0, grid_size[0], step_size)):
            for y_idx, y in enumerate(range(0, grid_size[0], step_size)):

                center_xy = [x + (step_size // 2), y + (step_size // 2)]

                if log_transform_displacements_bool:
                    x_disp = np.sign(x_y_displacements[x_idx, y_idx, 0]) * (
                        2 ** (abs(x_y_displacements[x_idx, y_idx, 0])) - 1
                    )
                    y_disp = np.sign(x_y_displacements[x_idx, y_idx, 1]) * (
                        2 ** (abs(x_y_displacements[x_idx, y_idx, 1])) - 1
                    )
                else:
                    x_disp = x_y_displacements[x_idx, y_idx, 0]
                    y_disp = x_y_displacements[x_idx, y_idx, 1]
                #  print(x_cent, y_cent, x_disp, y_disp)

                ax[1, 1].arrow(center_xy[0], center_xy[1], x_disp, y_disp)

        plt.show()

    # # ############################# end of DEBUGGING VISUALISATION #################

    return x_y_displacements, sub_class, displacement_weights
