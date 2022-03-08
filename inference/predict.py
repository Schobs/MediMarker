# from __future__ import print_function, absolute_import
import argparse
import os
import numpy as np
import sys

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything
import kale.utils.logger as logging
from inference.predict import get_coords

from dataset import ASPIRELandmarks

import csv

def predict(model_path)


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
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    logger = logging.construct_logger("aspire landmark U-Net", cfg.OUTPUT_DIR)
    logger.info(f"Using {device}")
    logger.info("\n" + cfg.dump())
    # ---- setup dataset ----



    #

    print(cfg.DATASET.ROOT,  cfg.DATASET.SRC_TARGETS, "AND ", os.path.join(cfg.DATASET.ROOT, cfg.DATASET.SRC_TARGETS))
    test_dataset = ASPIRELandmarks(
        annotation_path =cfg.DATASET.SRC_TARGETS,
        landmarks = cfg.DATASET.LANDMARKS,
        split = "testing",
        root_path = cfg.DATASET.ROOT,
        sigma = cfg.MODEL.GAUSS_SIGMA,
        cv = 1,
        cache_data = True,
        normalize=True,
        num_res_supervisions = cfg.SOLVER.NUM_RES_SUPERVISIONS 

    )

    test_dataloader = DataLoader(test_dataset, batch_size=cfg.SOLVER.DATA_LOADER_BATCH_SIZE, shuffle=False)


    #Trainer 
    trainer = UnetTrainer(1, train_dataloader, valid_dataloader, model_config= cfg, output_folder=cfg.OUTPUT_DIR)
    trainer.initialize()

    trainer.train()
    # for i_batch, sample in enumerate(dataloader):
        # print(i_batch, image.shape, label.shape)

if __name__ == "__main__":
    main()
