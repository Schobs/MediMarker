"""
Default configurations for action recognition domain adaptation. For full
selection of config options, please refer to the documentation.

Author: Lawrence Schobs, Ethan Jones
Last updated: 2023-01-05
"""

import os
from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()

# -----------------------------------------------------------------------------
# Dataset settings
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.DATASET_CLASS = 'generic'
_C.DATASET.ROOT =''
_C.DATASET.NAME = "ASPIRE"
_C.DATASET.SRC_TARGETS = '/shared/tale2/Shared/data/CMRI/ASPIRE/cardiac4ch_labels_VPnC_CV'
_C.DATASET.IMAGE_MODALITY= 'CMRI'
_C.DATASET.LANDMARKS = []
_C.DATASET.TRAINSET_SIZE = -1

# -----------------------------------------------------------------------------
# Sampler settings
# -----------------------------------------------------------------------------
_C.SAMPLER = CN()
_C.SAMPLER.SAMPLE_MODE = 'full' # ['patch', 'full']
_C.SAMPLER.DEBUG = False
_C.SAMPLER.INPUT_SIZE = [512,512]
_C.SAMPLER.DATA_AUG = 'AffineComplex' # ['AffineComplex']
_C.SAMPLER.DATA_AUG_PACKAGE = 'imgaug' #['imgaug', 'albumentations']
_C.SAMPLER.NUM_WORKERS = 0
_C.SAMPLER.PATCH = CN() 
_C.SAMPLER.PATCH.RESOLUTION_TO_SAMPLE_FROM = "full" # ['full', 'input_size']
_C.SAMPLER.PATCH.SAMPLE_PATCH_SIZE = [512,512]
_C.SAMPLER.PATCH.SAMPLER_BIAS = 0.66
_C.SAMPLER.PATCH.INFERENCE_MODE = "fully_convolutional" #["patchify_and_stitch", "fully_convolutional"] # patchify_and_stitch if you wish to patchify and stitch the RESOLUTION_TO_SAMPLE_FROM sized image, fully_convolutional to use input size directly RESOLUTION_TO_SAMPLE_FROM

# ----------------------------------------------------------------------------
# Solver settings
# ----------------------------------------------------------------------------
_C.SOLVER = CN()
_C.SOLVER.SEED = None
_C.SOLVER.BASE_LR = 0.01  # Initial learning rate
_C.SOLVER.DECAY_POLICY = "poly" # ["poly", None]
_C.SOLVER.MAX_EPOCHS = 1000 
_C.SOLVER.MINI_BATCH_SIZE = 150 
_C.SOLVER.DATA_LOADER_BATCH_SIZE = 12 
_C.SOLVER.NUM_RES_SUPERVISIONS = 5 
_C.SOLVER.AUTO_MIXED_PRECISION = True
_C.SOLVER.DEEP_SUPERVISION = True
_C.SOLVER.LOSS_FUNCTION = "mse" # # ["mse", "awl"]
_C.SOLVER.REGRESS_SIGMA = True #For sigma regess...
_C.SOLVER.REGRESS_SIGMA_LOSS_WEIGHT = 0.005

# ----------------------------------------------------------------------------
# Model trainer settings
# ----------------------------------------------------------------------------
_C.TRAINER = CN()
_C.TRAINER.PERFORM_VALIDATION = True
_C.TRAINER.SAVE_LATEST_ONLY = True 
_C.TRAINER.CACHE_DATA = True
_C.TRAINER.FOLD = 0
_C.TRAINER.INFERENCE_ONLY = False
_C.TRAINER.MCDROP_RATE = 0.0 #Dropout rate

# ----------------------------------------------------------------------------
# Model settings settings
# ----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.ARCHITECTURE = "U-Net"  # ["U-Net" , "PHD-Net"]
_C.MODEL.GAUSS_SIGMA = 4
_C.MODEL.HM_LAMBDA_SCALE = 100.0
_C.MODEL.CHECKPOINT= None

_C.MODEL.UNET = CN()
_C.MODEL.UNET.MIN_FEATURE_RESOLUTION = 4
_C.MODEL.UNET.MAX_FEATURES= 512
_C.MODEL.UNET.INIT_FEATURES= 32

_C.MODEL.PHDNET = CN()
_C.MODEL.PHDNET.BRANCH_SCHEME = 'multi' # ['multi', 'heatmap', 'displacement']
_C.MODEL.PHDNET.MAXPOOL_FACTOR = 3 # ['multi', 'heatmap', 'displacement']
_C.MODEL.PHDNET.CLASS_LABEL_SCHEME = "gaussian" # ['binary', 'binary_weighted', 'gaussian']
_C.MODEL.PHDNET.WEIGHT_DISP_LOSS_BY_HEATMAP = True # ['binary', 'binary_weighted', 'gaussian']
_C.MODEL.PHDNET.LOG_TRANSFORM_DISPLACEMENTS = True # ['binary', 'binary_weighted', 'gaussian']
_C.MODEL.PHDNET.CLAMP_DIST = None # [None, 48, 10] (none or any int. TODO: WRITE TEST FOR THIS)

# ----------------------------------------------------------------------------
# Inference settings
# ----------------------------------------------------------------------------
_C.INFERENCE = CN()
_C.INFERENCE.EVALUATION_MODE = "scale_heatmap_first" # ["scale_heatmap_first", "scale_pred_coords", "use_input_size"]
_C.INFERENCE.FIT_GAUSS = False # If false, uses max, if true, first fits gaussian to output heatmap.
_C.INFERENCE.ENSEMBLE_INFERENCE = False # average predictions from multiple models
_C.INFERENCE.TTA_ENSEMBLE_INFERENCE = False # average predictions from multiple augmented image samples
_C.INFERENCE.ENSEMBLE_UNCERTAINTY_KEYS = ["smha", "emha", "ecpv"] #keys for uncertainty estimation. 
_C.INFERENCE.UNCERTAINTY_SMHA_MODEL_IDX = 0 #keys for uncertainty estimation. 
_C.INFERENCE.ENSEMBLE_CHECKPOINTS = [] # list of checkpoints to ensemble
_C.INFERENCE.MCDROP_ENSEMBLE_INFERENCE = False  # average predictions from identical image samples using dropout layers
_C.INFERENCE.DEBUG = False

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT = CN()
_C.OUTPUT.VERBOSE = True 
_C.OUTPUT.OUTPUT_DIR = "/output/"
_C.OUTPUT.USE_COMETML_LOGGING = False 
_C.OUTPUT.COMET_API_KEY = None
_C.OUTPUT.COMET_WORKSPACE = "default"
_C.OUTPUT.COMET_PROJECT_NAME = "LannU-Net"
_C.OUTPUT.COMET_TAGS = ["default"]
_C.OUTPUT.RESULTS_CSV_APPEND = None

def get_cfg_defaults():
    return _C.clone()
