# from __future__ import print_function, absolute_import
import argparse
import os
import numpy as np
import sys

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.tensorboard import SummaryWriter


from dataset import ASPIRELandmarks
from config import get_cfg_defaults
from model_trainer import UnetTrainer

import csv



def arg_parse():
    parser = argparse.ArgumentParser(description="PyTorch Landmark Localization U-Net")
    parser.add_argument("--cfg", required=True, help="path to config file", type=str)
    parser.add_argument(
        "--gpus",
        default=1,
        help="gpu id(s) to use. None/int(0) for cpu. list[x,y] for xth, yth GPU."
        "str(x) for the first x GPUs. str(-1)/int(-1) for all available GPUs",
    )
    parser.add_argument("--resume", default="", type=str)
    args = parser.parse_args()
    return args

def main():
    """The main for this domain adaptation example, showing the workflow"""
    args = arg_parse()


    # ---- setup device ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("==> Using device " + device)

    # ---- setup configs ----
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    print(cfg)

    seed = cfg.SOLVER.SEED
    seed_everything(seed)


    # ---- setup output ----
    # os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # logger = logging.construct_logger("aspire landmark U-Net", cfg.OUTPUT_DIR)
    # logger.info(f"Using {device}")
    # logger.info("\n" + cfg.dump())

    # ---- setup dataset ----


    #  annotation_path: str,
    #     num_landmarks: int,
    #     patch_size: List[int],
    #     image_modality: str= "CMRI",
    #     split: str ="training",
    #     root_path: str = "./data",
    #     sigma: float = 3.0,
    #     cv: int = -1,
    #     cache_data: bool = False,
    #     normalize: bool = True

    #

    print(cfg.DATASET.ROOT,  cfg.DATASET.SRC_TARGETS, "AND ", os.path.join(cfg.DATASET.ROOT, cfg.DATASET.SRC_TARGETS))

    writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR))
    prof = torch.profiler.profile(
        on_trace_ready=torch.profiler.tensorboard_trace_handler(cfg.OUTPUT_DIR),
        record_shapes=True,
        with_stack=True)
    #Trainer 
    trainer = UnetTrainer(1, model_config= cfg, output_folder=cfg.OUTPUT_DIR, logger=writer)
    trainer.initialize(training_bool=True)

    trainer.train()
    # for i_batch, sample in enumerate(dataloader):
        # print(i_batch, image.shape, label.shape)

if __name__ == "__main__":
    main()
