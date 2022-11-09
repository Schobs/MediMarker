import argparse
from config import get_cfg_defaults  # pylint: disable=import-error
import warnings
from yacs.config import CfgNode as CN


def get_evaluation_mode(eval_mode):
    """_summary_

    Args:
        eval_mode (str): string for evaulation mode
        og_im_size [int, int]: original image size
        inp_size [int, int]: image resized for input to network (can be same as og_im_size)

    Raises:
        ValueError: if eval_mode is not supported

    Returns:
        bool, bool: settings for the evaluation modes.
    """

    # Evaluate on input size to network, using coordinates resized to the input size
    if eval_mode == "use_input_size":
        use_full_res_coords = False
        resize_first = False
    # Scale model predicted sized heatmap up to full resolution and then obtain coordinates (recommended)
    elif eval_mode == "scale_heatmap_first":
        use_full_res_coords = True
        resize_first = True
    # Obtain coordinates from input sized heatmap and scale up the coordinates to full sized heatmap.
    elif eval_mode == "scale_pred_coords":
        use_full_res_coords = True
        resize_first = False
    else:
        raise ValueError(
            "value for cg.INFERENCE.EVALUATION_MODE not recognised. Choose from: scale_heatmap_first, scale_pred_coords, use_input_size"
        )
    return use_full_res_coords, resize_first


def infer_additional_arguments(yaml_args):
    yaml_args.INFERRED_ARGS = CN()
    yaml_args.INFERRED_ARGS.GEN_HM_IN_MAINTHREAD = False

    # due to multithreading issue, we must generate heatmap labels in the main thread rather than
    # multi-thread dataloaders. To fix this in future.
    if yaml_args.SAMPLER.NUM_WORKERS != 0 and yaml_args.SOLVER.REGRESS_SIGMA:
        yaml_args.INFERRED_ARGS.GEN_HM_IN_MAINTHREAD = True

    use_full_res_coords, resize_first = get_evaluation_mode(
        yaml_args.INFERENCE.EVALUATION_MODE
    )

    yaml_args.INFERRED_ARGS.USE_FULL_RES_COORDS = use_full_res_coords
    yaml_args.INFERRED_ARGS.RESIZE_FIRST = resize_first

    # If we are doing patch sampling and specify evaluation is on input_size, sample validation in patches. Otherwise, sample full image.
    if (
        yaml_args.SAMPLER.SAMPLE_MODE == "patch"
        and yaml_args.INFERENCE.EVALUATION_MODE == "use_input_size"
    ):
        yaml_args.INFERRED_ARGS.EVALUATION_SAMPLE_MODE = "patch"
    else:
        yaml_args.INFERRED_ARGS.EVALUATION_SAMPLE_MODE = "full"
    if yaml_args.SAMPLER.PATCH.RESOLUTION_TO_SAMPLE_FROM == "input_size":
        yaml_args.SAMPLER.PATCH.RESOLUTION_TO_SAMPLE_FROM = yaml_args.SAMPLER.INPUT_SIZE
    else:
        raise NotImplementedError(
            "Only input_size is supported for now for SAMPLER.PATCH.RESOLUTION_TO_SAMPLE_FROM, not full"
        )

    return yaml_args


def argument_checking(yaml_args):
    """Checks on arguments to make sure wrong combination of arguments can not be input.

    Args:
        yaml_args (yaml): yaml config file loaded as a varible

    Raises:
        ValueError: Errors based on argument combinations.
    """
    all_errors = []
    # if yaml_args.SAMPLER.NUM_WORKERS != 0 and yaml_args.SOLVER.REGRESS_SIGMA:

    #### Parse sampler arguments ####

    try:
        accepted_samplers = ["patchify_and_stitch", "fully_convolutional"]
        if yaml_args.SAMPLER.PATCH.INFERENCE_MODE not in accepted_samplers:
            raise ValueError(
                "SAMPLER.SAMPLE_MODE %s not recognised. Choose from %s"
                % (yaml_args.SAMPLER.SAMPLE_MODE, accepted_samplers)
            )
    except ValueError as e:
        all_errors.append(e)

    # temp for not implemented yet
    try:
        if yaml_args.SAMPLER.PATCH.INFERENCE_MODE == "patchify_and_stitch":
            raise ValueError(
                'SAMPLER.PATCH.INFERENCE_MODE "patchify_and_stitch" not implemented yet.'
            )
    except ValueError as e:
        all_errors.append(e)

    try:
        accepted_samplers = ["patch", "full"]
        if yaml_args.SAMPLER.SAMPLE_MODE not in accepted_samplers:
            raise ValueError(
                "SAMPLER.SAMPLE_MODE %s not recognised. Choose from %s"
                % (yaml_args.SAMPLER.SAMPLE_MODE, accepted_samplers)
            )
    except ValueError as e:
        all_errors.append(e)

    # Patch sampling cases to cover:
    # 1) General:
    #   a) ensure SAMPLER.PATCH.SAMPLE_PATCH_SIZE !=  DATASET.INPUT_SIZE if resizing images to input_size
    #   b) ensure SAMPLER.PATCH.SAMPLE_PATCH_SIZE < DATASET.INPUT_SIZE if resizing images to input_size
    #   c) If user is sampling from full resolution, provide a warning that the patch size should be smaller than the image size

    if yaml_args.SAMPLER.SAMPLE_MODE == "patch":

        # 1a
        try:
            if (
                yaml_args.SAMPLER.PATCH.RESOLUTION_TO_SAMPLE_FROM == "input_size"
                and yaml_args.SAMPLER.PATCH.SAMPLE_PATCH_SIZE
                == yaml_args.SAMPLER.INPUT_SIZE
            ):
                raise ValueError(
                    """You want to train the model by sampling patches (SAMPLER.SAMPLE_MODE == "patch") from the image resized to input size defined by """
                    """SAMPLER.INPUT_SIZE, indicated by SAMPLER.PATCH.RESOLUTION_TO_SAMPLE_FROM == "input_size"""
                    """However, your SAMPLER.SAMPLE_PATCH_SIZE (%s) is the same as your input size SAMPLER.INPUT_SIZE (%s). You are doing "full" sampling, """
                    """ unintentionally running the wrong scheme."""
                    % (yaml_args.SAMPLER.SAMPLE_PATCH, yaml_args.SAMPLER.INPUT_SIZE)
                )
        except ValueError as e:
            all_errors.append(e)

        # 1b
        try:
            if (
                yaml_args.SAMPLER.PATCH.RESOLUTION_TO_SAMPLE_FROM == "input_size"
                and yaml_args.SAMPLER.PATCH.SAMPLE_PATCH_SIZE
                > yaml_args.SAMPLER.INPUT_SIZE
            ):
                raise ValueError(
                    """ You are using SAMPLER.PATCH.RESOLUTION_TO_SAMPLE_FROM == "input_size"  and your SAMPLER.PATCH.SAMPLE_PATCH_SIZE is larger than the SAMPLER.INPUT_SIZE (%s vs %s)"""
                    """It must be smaller."""
                    % (
                        yaml_args.SAMPLER.PATCH.RESOLUTION_TO_SAMPLE_FROM,
                        yaml_args.SAMPLER.INPUT_SIZE,
                    )
                )
        except ValueError as e:
            all_errors.append(e)

        # 1c
        if yaml_args.SAMPLER.PATCH.RESOLUTION_TO_SAMPLE_FROM == "full":
            warnings.warn(
                'You are doing patch-bases sampling, sampling from full resolution images (SAMPLER.PATCH.RESOLUTION_TO_SAMPLE_FROM == "full").'
                "Make sure the SAMPLER.PATCH.SAMPLE_PATCH_SIZE is smaller than ALL of your input images! Future: make a pre-processing check for this. ",
                stacklevel=2,
            )

    # deep supervision cases to cover:
    try:
        if (
            yaml_args.MODEL.ARCHITECTURE != "U-Net"
            and yaml_args.SOLVER.DEEP_SUPERVISION
        ):
            raise ValueError(
                "Only MODEL.ARCHITECTURE U-Net can be used to deep supervision (SOLVER.DEEP_SUPERVISION is True). You are using ",
                yaml_args.MODEL.ARCHITECTURE,
            )
    except ValueError as e:
        all_errors.append(e)

    try:
        if (
            not yaml_args.SOLVER.DEEP_SUPERVISION
            and yaml_args.SOLVER.NUM_RES_SUPERVISIONS > 1
        ):
            raise ValueError(
                """(SOLVER.DEEP_SUPERVISION is False), but your number of resolution supervision levels (SOLVER.NUM_RES_SUPERVISIONS) is %s. """
                """ Please set to 1 or turn DEEP_SUPERVISION =True if you want to use >1 supervision level."""
                % yaml_args.SOLVER.NUM_RES_SUPERVISIONS
            )
    except ValueError as e:
        all_errors.append(e)

    try:
        if (
            yaml_args.SAMPLER.DATA_AUG != None
            and yaml_args.SAMPLER.DATA_AUG_PACKAGE != "imgaug"
        ):
            raise ValueError(
                "Only the imgaug data augmentation package (SAMPLER.DATA_AUG_PACKAGE) is supported, you chose %s. Try 'imgaug' or set SAMPLER.DATA_AUG to None for no data augmentation."
                % yaml_args.SAMPLER.DATA_AUG_PACKAGE
            )
    except ValueError as e:
        all_errors.append(e)

    # Warnings
    if (
        yaml_args.SOLVER.REGRESS_SIGMA
        and yaml_args.SOLVER.REGRESS_SIGMA_LOSS_WEIGHT <= 0
    ):
        warnings.warn(
            "You are attempting to regress sigmas (yaml_args.SOLVER.REGRESS_SIGMA=True) but your yaml_args.REGRESS_SIGMA_LOSS_WEIGHT is <=0."
            "This means the magnitude of sigma will not be penalized and could lead to trivial solution sigma -> inf."
            "Consider this setting this to a positive float (e.g. 0.005). You are warned! ",
            stacklevel=2,
        )

    if yaml_args.SAMPLER.DATA_AUG == None:
        warnings.warn(
            "You are not using data augmentation (SAMPLER.DATA_AUG = None). Using a data augmentation scheme will improve your results.",
            stacklevel=2,
        )

    if all_errors:
        print("I have identified some issues with your .yaml config file:")
        raise ValueError(all_errors)


def arg_parse():
    """Parses shell arguments, loads the default config file and merges it with user defined arguments. Calls the argument checker.
    Returns:
        config: config for the programme
    """
    parser = argparse.ArgumentParser(description="PyTorch Landmark Localization U-Net")
    parser.add_argument("--cfg", required=True, help="path to config file", type=str)
    parser.add_argument(
        "--gpus",
        default=1,
        help="gpu id(s) to use. None/int(0) for cpu. list[x,y] for xth, yth GPU."
        "str(x) for the first x GPUs. str(-1)/int(-1) for all available GPUs",
    )
    parser.add_argument("--fold", type=int)
    args = parser.parse_args()

    # ---- setup configs ----
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)

    if args.fold:
        cfg.TRAINER.FOLD = args.fold
    # cfg.freeze()
    cfg = infer_additional_arguments(cfg)

    print("Config: \n ", cfg)

    argument_checking(cfg)
    return cfg
