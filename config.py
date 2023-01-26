"""
Default configurations for action recognition domain adaptation
"""


from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.DATASET_CLASS = "generic"  # see datasets.dataset_index for available datasets or add new ones yourself.

_C.DATASET.ROOT = ""
_C.DATASET.NAME = "ASPIRE"
_C.DATASET.SRC_TARGETS = (
    "/shared/tale2/Shared/data/CMRI/ASPIRE/cardiac4ch_labels_VPnC_CV"
)
_C.DATASET.IMAGE_MODALITY = "CMRI"
_C.DATASET.LANDMARKS = []
# _C.DATASET.ORIGINAL_IMAGE_SIZE = [512,512] #legacy, can ignore.
_C.DATASET.TRAINSET_SIZE = -1  # -1 for full trainset size or int <= len(training_set)

# _C.DATASET.DEBUG = False
# _C.DATASET.DATA_AUG = None
# _C.DATASET.DATA_AUG_PACKAGE = 'imgaug' #['imgaug', 'albumentations']
# _C.DATASET.NUM_WORKERS = 8


_C.SAMPLER = CN()
_C.SAMPLER.SAMPLE_MODE = "full"  # ['full', "patch_bias", 'patch_centred']

_C.SAMPLER.DEBUG = False
_C.SAMPLER.INPUT_SIZE = [512, 512]
# ['patch', "full"] Whether to sample patches or the full image during evaluation
_C.SAMPLER.EVALUATION_SAMPLE_MODE = "full"

_C.SAMPLER.DATA_AUG = "AffineComplex"  # None
_C.SAMPLER.DATA_AUG_PACKAGE = "imgaug"  # ['imgaug', 'albumentations']
_C.SAMPLER.NUM_WORKERS = 0


_C.SAMPLER.PATCH = CN()
# ['full', 'input_size'] # if input_size, will resize image to SAMPLER.INPUT_SIZE first.
_C.SAMPLER.PATCH.RESOLUTION_TO_SAMPLE_FROM = "input_size"
_C.SAMPLER.PATCH.SAMPLE_PATCH_SIZE = [512, 512]
# ["patchify_and_stitch", "fully_convolutional"] # patchify_and_stitch if you wish to patchify and stitch the RESOLUTION_TO_SAMPLE_FROM sized image, fully_convolutional to use input size directly RESOLUTION_TO_SAMPLE_FROM
_C.SAMPLER.PATCH.INFERENCE_MODE = "fully_convolutional"

_C.SAMPLER.PATCH.SAMPLER_BIAS = 0.66

_C.SAMPLER.PATCH.CENTRED_PATCH_COORDINATE_PATH = None
_C.SAMPLER.PATCH.CENTRED_PATCH_COORDINATE_PATH_SHEET = None
_C.SAMPLER.PATCH.CENTRED_PATCH_JITTER = 0.0

# U:\tale2\Shared\data\CMRI\ASPIRE\4CH_labels
# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.SEED = None
_C.SOLVER.BASE_LR = 0.01  # Initial learning rate
# _C.SOLVER.MOMENTUM = 0.99
_C.SOLVER.DECAY_POLICY = "poly"  # ["poly", None]
# _C.SOLVER.NESTEROV = True
# _C.SOLVER.TYPE = "SGD"
_C.SOLVER.MAX_EPOCHS = 1000
_C.SOLVER.EARLY_STOPPING_PATIENCE = 150
_C.SOLVER.MINI_BATCH_SIZE = 150
_C.SOLVER.DATA_LOADER_BATCH_SIZE_TRAIN = 12
_C.SOLVER.DATA_LOADER_BATCH_SIZE_EVAL = 12


_C.SOLVER.NUM_RES_SUPERVISIONS = 5
_C.SOLVER.AUTO_MIXED_PRECISION = True
_C.SOLVER.DEEP_SUPERVISION = True
_C.SOLVER.LOSS_FUNCTION = "mse"  # # ["mse", "awl"]
_C.SOLVER.REGRESS_SIGMA = False
_C.SOLVER.REGRESS_SIGMA_LOSS_WEIGHT = 0.005


# model trainer
_C.TRAINER = CN()
_C.TRAINER.PERFORM_VALIDATION = True
_C.TRAINER.SAVE_LATEST_ONLY = True
_C.TRAINER.SAVE_EVERY = 10

_C.TRAINER.VALIDATE_EVERY = 10

_C.TRAINER.CACHE_DATA = True
_C.TRAINER.FOLD = 0
_C.TRAINER.INFERENCE_ONLY = True


# ---------------------------------------------------------------------------- #
# U-Net Model configs# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.ARCHITECTURE = "U-Net"  # ["U-Net" , "PHD-Net"]
# _C.MODEL.ACTIVATION_FUNCTION = "leaky_relu"
_C.MODEL.GAUSS_SIGMA = 4
_C.MODEL.HM_LAMBDA_SCALE = 100.0
# _C.MODEL.KERNEL_SIZE = 3
_C.MODEL.CHECKPOINT = None

_C.MODEL.UNET = CN()
_C.MODEL.UNET.MIN_FEATURE_RESOLUTION = 4
_C.MODEL.UNET.MAX_FEATURES = 512
_C.MODEL.UNET.INIT_FEATURES = 32


_C.MODEL.PHDNET = CN()
_C.MODEL.PHDNET.BRANCH_SCHEME = "multi"  # ['multi', 'heatmap', 'displacement']
_C.MODEL.PHDNET.MAXPOOL_FACTOR = 3  # ['multi', 'heatmap', 'displacement']
_C.MODEL.PHDNET.CLASS_LABEL_SCHEME = (
    "gaussian"  # ['binary', 'binary_weighted', 'gaussian']
)
_C.MODEL.PHDNET.WEIGHT_DISP_LOSS_BY_HEATMAP = True
_C.MODEL.PHDNET.LOG_TRANSFORM_DISPLACEMENTS = True

# [None, 48, 10] (none or any int. TODO: WRITE TEST FOR THIS)
_C.MODEL.PHDNET.CLAMP_DIST = None


_C.INFERENCE = CN()
_C.INFERENCE.SPLIT = "testing"  # Split/s of the data to perform inference. Can specify multiple splits.

_C.INFERENCE.EVALUATION_MODE = "scale_heatmap_first"  # ["scale_heatmap_first", "scale_pred_coords", "use_input_size"]
_C.INFERENCE.FIT_GAUSS = (
    False  # If false, uses max, if true, first fits gaussian to output heatmap.
)

_C.INFERENCE.ENSEMBLE_INFERENCE = False  # average predictions from multiple models
_C.INFERENCE.ENSEMBLE_UNCERTAINTY_KEYS = [
    "smha",
    "emha",
    "ecpv",
]  # keys for uncertainty estimation.
_C.INFERENCE.UNCERTAINTY_SMHA_MODEL_IDX = 0  # keys for uncertainty estimation.

_C.INFERENCE.ENSEMBLE_CHECKPOINTS = []  # list of checkpoints to ensemble

_C.INFERENCE.DEBUG = False
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT = CN()
_C.OUTPUT.VERBOSE = True
# _C.OUTPUT.FAST_DEV_RUN = False  # True for debug
_C.OUTPUT.OUTPUT_DIR = "/output/"

# _C.OUTPUT.TB_DIR = "./lightning_logs"

_C.OUTPUT.USE_COMETML_LOGGING = False

_C.OUTPUT.COMET_API_KEY = None
_C.OUTPUT.COMET_WORKSPACE = "default"
_C.OUTPUT.COMET_PROJECT_NAME = "landnnunet"

_C.OUTPUT.COMET_TAGS = ["default"]
_C.OUTPUT.RESULTS_CSV_APPEND = None

# ---------------------------------------------------------------------------- #
# SSH options
# -----------
# Use if your data is on an SSH mount and you want to load data & save results to
# this mount (common if you are using HPC). Only use this if you understand it.
# If set to True, set your SRC_TARGETS, OUTPUT_DIR AND ROOT to the path you use
# when you are running the programme inside your SSH mount.
# Then set your LOCAL_PATH_TO_SSH_MOUNT to the path from your local machine to
# the SSH mount. The code will automatically detect if you are running locally
# and alter your paths to the SSH mount accordingly.
# Example:
# DATASET.ROOT: "/shared/tale2/Shared/schobs/data/ISBI2015_landmarks"
# SSH.LOCAL_PATH_TO_SSH_MOUNT = "/mnt/tale_shared/"
# and the code will detect if you are running locally instead of via SSH and change:
# DATASET.ROOT = "/mnt/tale_shared/shared/tale2/Shared/schobs/data/ISBI2015_landmarks"
# ---------------------------------------------------------------------------- #

_C.SSH = CN()
_C.SSH.AUTO_AMEND_PATHS = True
_C.SSH.LOCAL_PATH_TO_SSH_MOUNT = "/mnt/bess"
_C.SSH.MATCH_SSH_STRING_FOR_AMEND = '/shared/tale2/'


def get_cfg_defaults():
    return _C.clone()
