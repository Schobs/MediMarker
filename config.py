"""
Default configurations for action recognition domain adaptation
"""

import os

from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.ROOT ='/shared/tale2/Shared/data/CMRI/ASPIRE'
_C.DATASET.NAME = "ASPIRE"
_C.DATASET.SRC_TARGETS = '/shared/tale2/Shared/data/CMRI/ASPIRE/cardiac4ch_labels_VPnC_CV'
_C.DATASET.IMAGE_MODALITY= 'CMRI'
_C.DATASET.LANDMARKS = [0,1,2]
_C.DATASET.ORIGINAL_IMAGE_SIZE = [512,512]
_C.DATASET.INPUT_SIZE = [512,512]

_C.DATASET.DEBUG = False
_C.DATASET.DATA_AUG = None


# U:\tale2\Shared\data\CMRI\ASPIRE\4CH_labels
# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.SEED = 42
_C.SOLVER.BASE_LR = 0.01  # Initial learning rate
_C.SOLVER.MOMENTUM = 0.99
_C.SOLVER.DECAY_POLICY = "poly"
_C.SOLVER.NESTEROV = True
_C.SOLVER.TYPE = "SGD"
_C.SOLVER.MAX_EPOCHS = 1000 
_C.SOLVER.MINI_BATCH_SIZE = 250 
_C.SOLVER.DATA_LOADER_BATCH_SIZE = 12 
_C.SOLVER.NUM_RES_SUPERVISIONS = 5 
_C.SOLVER.AUTO_MIXED_PRECISION = True
_C.SOLVER.DEEP_SUPERVISION = True
_C.SOLVER.LOSS_FUNCTION = "mse"


# ---------------------------------------------------------------------------- #
# U-Net Model configs# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.ARCHITECTURE = "U-Net"  
_C.MODEL.MIN_FEATURE_RESOLUTION = 4
_C.MODEL.ACTIVATION_FUNCTION = "leaky_relu"
_C.MODEL.KERNEL_SIZE = 3
_C.MODEL.GAUSS_SIGMA = 4
_C.MODEL.MAX_FEATURES= 512
_C.MODEL.INIT_FEATURES= 32
_C.MODEL.CHECKPOINT= None





# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT = CN()
_C.OUTPUT.VERBOSE = True 
_C.OUTPUT.FAST_DEV_RUN = False  # True for debug
_C.OUTPUT_DIR = "/shared/tale2/Shared/schobs/landmark_unet/ensemble/outputs/template"
_C.OUTPUT.TB_DIR = "./lightning_logs"

def get_cfg_defaults():
    return _C.clone()
