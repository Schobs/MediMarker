# from __future__ import print_function, absolute_import
import argparse
import os
import numpy as np
import sys

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything
# import kale.utils.logger as logging


from dataset import ASPIRELandmarks
from config import get_cfg_defaults
from model_trainer import UnetTrainer

from visualisation import visualize_predicted_heatmaps
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



    # ---- setup dataset ----



    test_dataset = ASPIRELandmarks(
        annotation_path =cfg.DATASET.SRC_TARGETS,
        landmarks = cfg.DATASET.LANDMARKS,
        split = "training",
        root_path = cfg.DATASET.ROOT,
        sigma = cfg.MODEL.GAUSS_SIGMA,
        cv = 1,
        cache_data = True,
        normalize=True,
        num_res_supervisions = cfg.SOLVER.NUM_RES_SUPERVISIONS,
        original_image_size= cfg.DATASET.ORIGINAL_IMAGE_SIZE,
        input_size =  cfg.DATASET.INPUT_SIZE
 

    )

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


    #Load Model
    model_path = os.path.join(cfg.OUTPUT_DIR, "model_best_valid.model")
    # model_path = "../outputs/model_latest.model"

    trainer = UnetTrainer(1, model_config= cfg, output_folder=cfg.OUTPUT_DIR)
    trainer.load_checkpoint(model_path, training_bool=False)

    for i, data_dict in enumerate(test_dataloader):
     
        targ_coords =data_dict['target_coords'].cpu().detach().numpy()

        pred_heatmaps, pred_coords, loss = trainer.predict_heatmaps_and_coordinates(data_dict, return_all_layers=True)

        if isinstance(pred_heatmaps, list):
            pred_heatmaps = [x.cpu().detach().numpy() for x in pred_heatmaps]
        else:
            pred_heatmaps = pred_heatmaps.cpu().detach().numpy()
            
        pred_coords = pred_coords.cpu().detach().numpy()

        # print("pred heatmaps len: ",len(pred_heatmaps))

        # print("pred cooords: ", pred_coords.shape, pred_coords)
        # print("target coords: ", targ_coords.shape, targ_coords)
        coord_error = np.linalg.norm((pred_coords- targ_coords), axis=2)

        print("target Coords %s, predicted coords %s and error: %s. Loss: %s " % (targ_coords, pred_coords, coord_error, np.mean(loss)))

        visualize_predicted_heatmaps(pred_heatmaps, pred_coords, targ_coords)

if __name__ == "__main__":
    main()
