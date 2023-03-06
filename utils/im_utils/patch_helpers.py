from ast import List, Tuple
import json
import numpy as np
from utils.im_utils.visualisation import visualize_centre_patch, visualize_patch
import logging


def sample_patch_with_bias(image, landmarks, sample_patch_bias, load_im_size,  sample_patch_size, logger, debug, lm_safe_region=0, safe_padding=128):
    """Samples a patch from the image. It ensures a landmark is in a patch with a self.sample_patch_bias% chance.
        The patch image is larger than the patch-size by safe_padding on every side for safer data augmentation.
        Therefore, the image is first padded with zeros on each side to stop out of bounds when sampling from the edges.

    Args:
        image (_type_): image to sample
        landmarks (_type_): list of landmarks
        # pixels away from the edge the landmark must be to count as "in" the patch . Defaults to 0.
        lm_safe_region (int, optional):
        safe_padding (int, optional): How much bigger on each edge the patch should be for safer data augmentation . Defaults to 128.

    Returns:
        _type_: cropped padded sample
        landmarks normalised to within the patch
        binary indicator of which landmarks are in the patch.

    """

    logger = logging.getLogger()
    z_rand = np.random.uniform(0, 1)
    landmarks_in_indicator = []
    if z_rand >= (1 - sample_patch_bias):

        # Keep sampling until landmark is in patch
        while 1 not in landmarks_in_indicator:
            landmarks_in_indicator = []

            #
            y_rand = np.random.randint(
                0, load_im_size[1] - sample_patch_size[1]
            )
            x_rand = np.random.randint(
                0, load_im_size[0] - sample_patch_size[0]
            )

            for lm in landmarks:
                landmark_in = 0

                # Safe region means landmark is not right on the edge
                if (
                    y_rand + lm_safe_region
                    <= lm[1]
                    <= (y_rand + sample_patch_size[1]) - lm_safe_region
                ):
                    if (
                        x_rand + lm_safe_region
                        <= lm[0]
                        <= (x_rand + sample_patch_size[0]) - lm_safe_region
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
            0, load_im_size[1] - sample_patch_size[1]
        )
        x_rand = np.random.randint(
            0, load_im_size[0] - sample_patch_size[0]
        )

        for lm in landmarks:
            landmark_in = 0
            if (
                y_rand + lm_safe_region
                <= lm[1]
                <= y_rand + sample_patch_size[1] - lm_safe_region
            ):
                if (
                    x_rand + lm_safe_region
                    <= lm[0]
                    <= (x_rand + sample_patch_size[0]) - lm_safe_region
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
    padded_patch_size = [x + (2 * safe_padding) for x in sample_patch_size]

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

    if debug:
        padded_lm = [
            [lm[0] + safe_padding, lm[1] + safe_padding] for lm in landmarks
        ]

        logger.info(
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


def sample_patch_centred(image, coords_to_centre_around, load_im_size, sample_patch_size, centre_patch_jitter, debug, groundtruth_lms=None, safe_padding=128, deterministic=False, garuntee_gt_in=True):
    """ Samples a patch from an image, centred around coords_to_centre_around.
        Patch is padded with zeros to the size of the patch to allow fo future data augmentation.
        If the coordinate is too near the edge of the image, the patch is adjusted towards the centre.
        If centre_patch_jitter > 0, the patch is sampled randomly around the coords_to_centre_around (TODO).
        If groundtruth_lms is provided, the landmarks are adjusted to the new patch origin.


    Args:
        image (_type_): _description_
        coords_to_centre_around (_type_): _description_
        load_im_size (_type_): _description_
        sample_patch_size (_type_): _description_
        centre_patch_jitter (_type_): _description_
        debug (_type_): _description_
        groundtruth_lms (_type_, optional): _description_. Defaults to None.
        lm_safe_region (int, optional): _description_. Defaults to 0.
        safe_padding (int, optional): _description_. Defaults to 128.

    Raises:
        NotImplementedError: _description_

    Returns:
        _type_: _description_
    """
    logger = logging.getLogger()

    assert len(coords_to_centre_around) == 1, "Only one landmark to centre sampled patch around is supported"
    assert 0 <= centre_patch_jitter <= 1.0, "centre_patch_jitter must be between 0 and 1 (A proportion of the patch)."
    if groundtruth_lms is not None:
        assert len(groundtruth_lms) == len(
            coords_to_centre_around), "Number of landmarks must match number of landmarks to centre around (i.e. 1)"

    centre_coord = coords_to_centre_around[0]

    # Do the jitter around the centre landmark
    # TODO: jitter should be a float between 0-1. This will be the percentage of the patch size to jitter by.
    # Is % of image better than exact pixel # to jitter by? For now, it will be %
    # Pick a random direction in x and y, and move centre by that much.
    # Then the image edge code will handle if centre is off the image.
    landmarks_in_indicator = 0
    fail_count = 0
    while landmarks_in_indicator == 0:
        if fail_count > 10:
            raise ValueError("Failed to sample a patch with a landmark in it.")
        if centre_patch_jitter > 0:
            # if garuntee_gt_in:

            # Jitter needs to ensure centre_coord is still in the patch. Add some pixels for safety of image edges.
            max_x_jitter = max(0, ((sample_patch_size[0]/2)*centre_patch_jitter) - 5)
            max_y_jitter = max(0, ((sample_patch_size[1]/2)*centre_patch_jitter) - 5)

            # Ensure the same jitter is applied to the landmarks
            if deterministic:
                np.random.seed(int(coords_to_centre_around[0, 0] + coords_to_centre_around[0, 1]))

            x_dir = np.random.choice([-1, 1])
            y_dir = np.random.choice([-1, 1])
            x_jitter = x_dir * np.random.randint(0, max_x_jitter)
            y_jitter = y_dir * np.random.randint(0, max_y_jitter)

            centre_coord = [centre_coord[0] + x_jitter, centre_coord[1] + y_jitter]

            # raise NotImplementedError("Jitter not implemented for centre patch sampling")

        # Need to cover case where the landmark is near the edge of the image.
        if centre_coord[0] > (load_im_size[0]-(sample_patch_size[0]/2)):
            x_min = load_im_size[0]-sample_patch_size[0]
        else:
            x_min = centre_coord[0] - (sample_patch_size[0]/2)

        if centre_coord[1] > (load_im_size[1]-(sample_patch_size[1]/2)):
            y_min = load_im_size[1]-sample_patch_size[1]
        else:
            y_min = centre_coord[1] - (sample_patch_size[1]/2)

        # Cover the case where the landmark is near the close edge of the image.
        x_min = max(0, x_min)
        y_min = max(0, y_min)

        # check if the GT landmark is in the patch
        if groundtruth_lms is not None:
            if groundtruth_lms[0][0] < x_min or groundtruth_lms[0][0] > x_min + sample_patch_size[0] or groundtruth_lms[0][1] < y_min or groundtruth_lms[0][1] > y_min + sample_patch_size[1]:
                landmarks_in_indicator = 0
            else:
                landmarks_in_indicator = 1

        if not garuntee_gt_in:
            break

    patch_image = image[:, int(y_min):int(y_min+sample_patch_size[1]), int(x_min):int(x_min+sample_patch_size[0])]
    if groundtruth_lms is not None:
        # Calculate the new origin: 2*safe_padding bc we padded image & then added pad to the patch.
        normalized_landmarks = [
            [
                groundtruth_lms[0][0] - (x_min),
                groundtruth_lms[0][1] - (y_min),
            ]
        ]
    # If not provided, just return 0,0
    else:
        normalized_landmarks = [[0, 0]]

    normalized_centre_coords = [
        [
            centre_coord[0] - (x_min),
            centre_coord[1] - (y_min),
        ]
    ]

    if debug:

        if centre_patch_jitter > 0:

            logger.info("The original center coord passed is: %s,", coords_to_centre_around[0])
            logger.info("The jitter level %s, and x_jitter and y_jitter is: %s, %s.",
                        centre_patch_jitter, x_jitter, y_jitter)
            logger.info("The new centre coord is: %s ", centre_coord)
            logger.info("The sample patch size is: %s, so the x_min and y_min is: %s, %s ",
                        sample_patch_size, x_min, y_min)
            logger.info("The target lm is %s so the landmark_in indicator is: %s ",
                        groundtruth_lms[0], landmarks_in_indicator,)
            # logger.info("After we pad the image, the new x_rand_safe and y_rand_safe is: %s, %s  ",
            #             x_rand_safe, y_rand_safe)
            # logger.info("the new x_rand_pad and y_rand_pad is: %s, %s ", x_rand_pad, y_rand_pad)

            logger.info("The normalized centre coords are: % s ", normalized_centre_coords)
            logger.info("The normalized landmark is : % s", normalized_landmarks)

        # full_landmarks include: original center, jittered center, full groundtruth
        # padded_landmarks include : padded original center,, full groundtruth, jittered padded center
        # patch_landmarks include : original_norm_center, jittered_normalized center coords, normalized GT
        visualize_centre_patch(
            full_im=image[0],
            full_landmarks=[coords_to_centre_around[0],  groundtruth_lms[0], centre_coord],
            patch_landmarks=[coords_to_centre_around[0] - [x_min, y_min],
                             normalized_landmarks[0], normalized_centre_coords[0]],
            image_patch=patch_image[0],

        )

    return (
        patch_image,
        normalized_landmarks,
        landmarks_in_indicator,
        [x_min, y_min],
    )


def get_patch_stitching_info(image_size, patch_size):
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
