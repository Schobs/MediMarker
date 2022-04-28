

from calendar import c
import yaml
import argparse
from config import get_cfg_defaults
import warnings
from yacs.config import CfgNode as CN

def get_evaluation_mode(eval_mode, og_im_size, inp_size):
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
    if eval_mode == 'use_input_size':
        use_full_res_coords =False
        resize_first = False
    # Force "use_input_size" settings if the input size and og image size are the same
    elif og_im_size == inp_size: 
        print("your DATASET.ORIGINAL_IMAGE_SIZE == DATASET.INPUT_SIZE, therefore defaulting evaluation mode to \"use_input_size\"")
        use_full_res_coords =False
        resize_first = False
    # Scale model predicted sized heatmap up to full resolution and then obtain coordinates (recommended)
    elif eval_mode== 'scale_heatmap_first':
        use_full_res_coords =True
        resize_first = True
    # Obtain coordinates from input sized heatmap and scale up the coordinates to full sized heatmap.
    elif eval_mode == 'scale_pred_coords':
        use_full_res_coords =True
        resize_first = False
    else:
        raise ValueError("value for cg.INFERENCE.EVALUATION_MODE not recognised. Choose from: scale_heatmap_first, scale_pred_coords, use_input_size")
    return use_full_res_coords, resize_first

def infer_additional_arguments(yaml_args):
    yaml_args.INFERRED_ARGS = CN()
    yaml_args.INFERRED_ARGS.GEN_HM_IN_MAINTHREAD = False

    #due to multithreading issue, we must genreate heatmap labels in the main thread rather than
    # multi-thread dataloaders. To fix this in future.
    if yaml_args.DATASET.NUM_WORKERS != 0 and yaml_args.SOLVER.REGRESS_SIGMA:
        yaml_args.INFERRED_ARGS.GEN_HM_IN_MAINTHREAD = True

    use_full_res_coords, resize_first = get_evaluation_mode(yaml_args.INFERENCE.EVALUATION_MODE, yaml_args.DATASET.ORIGINAL_IMAGE_SIZE, yaml_args.DATASET.INPUT_SIZE)
    
    yaml_args.INFERRED_ARGS.USE_FULL_RES_COORDS = use_full_res_coords
    yaml_args.INFERRED_ARGS.RESIZE_FIRST = resize_first

    return yaml_args
def argument_checking(yaml_args):
    """ Checks on arguments to make sure wrong combination of arguments can not be input.

    Args:
        yaml_args (yaml): yaml config file loaded as a varible

    Raises:
        ValueError: Errors based on argument combinations.
    """
    all_errors = []

   

    try:
        if yaml_args.MODEL.ARCHITECTURE != "U-Net" and yaml_args.SOLVER.DEEP_SUPERVISION:
            raise ValueError("Only MODEL.ARCHITECTURE U-Net can be used to deep supervision (SOLVER.DEEP_SUPERVISION is True). You are using ", yaml_args.MODEL.ARCHITECTURE )
    except ValueError as e:
        all_errors.append(e)



    try:
        if yaml_args.DATASET.ORIGINAL_IMAGE_SIZE[0] < yaml_args.DATASET.INPUT_SIZE[0] and yaml_args.DATASET.ORIGINAL_IMAGE_SIZE[1] < yaml_args.DATASET.INPUT_SIZE[1]:
            raise (ValueError("DATASET.ORIGINAL_IMAGE_SIZE is smaller than  than the input size to the network (DATASET.INPUT_SIZE). Change input size to equal or smaller size of original image."))
    except ValueError as e:
        all_errors.append(e)

    # try:
    #     if yaml_args.DATASET.NUM_WORKERS != 0 and yaml_args.SOLVER.REGRESS_SIGMA:
    #         raise ValueError("Regressing sigmas (SOLVER.REGRESS_SIGMA=True) requires DATASET.NUM_WORKERS=0 due to multithreading issues with updating dataset object with new sigmas. You are attempting to use %s workers " % yaml_args.DATASET.NUM_WORKERS )
    # except ValueError as e:
    #     all_errors.append(e)
    

    try:
        if yaml_args.DATASET.DATA_AUG != None and yaml_args.DATASET.DATA_AUG_PACKAGE != "imgaug":
            raise ValueError("Only the imgaug data augmentation package (DATASET.DATA_AUG_PACKAGE) is supported, you chose %s. Try \'imgaug\' or set DATASET.DATA_AUG to None for no data augmentation." % yaml_args.DATASET.DATA_AUG_PACKAGE )
    except ValueError as e:
        all_errors.append(e)


    #Warnings
    if yaml_args.SOLVER.REGRESS_SIGMA and yaml_args.SOLVER.REGRESS_SIGMA_LOSS_WEIGHT <= 0:
        warnings.warn('You are attempting to regress sigmas (yaml_args.SOLVER.REGRESS_SIGMA=True) but your yaml_args.REGRESS_SIGMA_LOSS_WEIGHT is <=0. \
            This means the magnitude of sigma will not be penalized and could lead to trivial solution sigma -> inf. \
                Consider this setting this to a positive float (e.g. 0.005). You are warned! ', stacklevel=2)

    if yaml_args.DATASET.DATA_AUG == None:
        warnings.warn('You are not using data augmentation (DATASET.DATA_AUG = None). Using a data augmentation scheme will improve your results.', stacklevel=2)


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
    parser.add_argument("--fold", default=0, type=int)
    args = parser.parse_args()

    # ---- setup configs ----
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.TRAINER.FOLD = args.fold
    # cfg.freeze()
    cfg = infer_additional_arguments(cfg)

    print("Config: \n ", cfg)



    argument_checking(cfg)
    return cfg