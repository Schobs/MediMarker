# from __future__ import print_function, absolute_import
import argparse
import enum
import os
import numpy as np
import sys

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything
# import kale.utils.logger as logging


from datasets.dataset import DatasetBase
from config import get_cfg_defaults
from trainer.model_trainer_unet import UnetTrainer

from visualisation import visualize_predicted_heatmaps
import csv




def run_inference_model(logger, cfg, model_path, model_name):

    test_dataset = DatasetBase(
        annotation_path =cfg.DATASET.SRC_TARGETS,
        landmarks = cfg.DATASET.LANDMARKS,
        split = "validation",
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


    

    trainer = UnetTrainer(1, model_config= cfg, output_folder=cfg.OUTPUT_DIR)
    trainer.load_checkpoint(model_path, training_bool=False)
    
    all_errors = []
    landmark_errors = [[] for x in range(len(cfg.DATASET.LANDMARKS)) ]

    if cfg.INFERENCE.EVALUATION_MODE == 'use_input_size' or  cfg.DATASET.ORIGINAL_IMAGE_SIZE == cfg.DATASET.INPUT_SIZE:
        use_full_res_coords =False
        resize_first = False
    elif cfg.INFERENCE.EVALUATION_MODE == 'scale_heatmap_first':
        use_full_res_coords =True
        resize_first = True
    elif cfg.INFERENCE.EVALUATION_MODE == 'scale_pred_coords':
        use_full_res_coords =True
        resize_first = False
    else:
        raise ValueError("value for cg.INFERENCE.EVALUATION_MODE not recognised. Choose from: scale_heatmap_first, scale_pred_coords, use_input_size")
    
    for i, data_dict in enumerate(test_dataloader):
        

        if use_full_res_coords:
            targ_coords =data_dict['full_res_coords'].cpu().detach().numpy()
        else:
            targ_coords =data_dict['target_coords'].cpu().detach().numpy()


            

        pred_heatmaps, final_pred_heatmap, pred_coords, loss = trainer.predict_heatmaps_and_coordinates(data_dict, return_all_layers=True, resize_to_og=resize_first)    
        pred_coords = pred_coords.cpu().detach().numpy()


        #If we don't resize the heatmap first and we want to evaluate on full image size, we need to scale coordinates up.
        if use_full_res_coords and not resize_first :
            downscale_factor = [cfg.DATASET.ORIGINAL_IMAGE_SIZE[0]/cfg.DATASET.INPUT_SIZE[0], cfg.DATASET.ORIGINAL_IMAGE_SIZE[1]/cfg.DATASET.INPUT_SIZE[1]]
            pred_coords = pred_coords * downscale_factor

        coord_error = np.linalg.norm((pred_coords- targ_coords), axis=2)

        for sample in coord_error:
            for idx, er in enumerate(sample):
                all_errors.append(er)
                landmark_errors[idx].append(er)



        if cfg.DATASET.DEBUG:

            pred_heatmaps.append(final_pred_heatmap)
            if isinstance(pred_heatmaps, list):
                pred_heatmaps = [x.cpu().detach().numpy() for x in pred_heatmaps]
            else:
                pred_heatmaps = pred_heatmaps.cpu().detach().numpy()
            print("target Coords %s, predicted coords %s and error: %s. Loss: %s " % (targ_coords, pred_coords, coord_error, np.mean(loss)))
            visualize_predicted_heatmaps(pred_heatmaps, pred_coords, targ_coords)


    print("Total Mean Error %s +/- %s " % (np.mean(all_errors), np.std(all_errors)))
    for lm in range(len(landmark_errors)):
        print("Landmark %s Mean Error %s +/- %s " % (lm, np.mean(landmark_errors[lm]), np.std(landmark_errors[lm])))
        logger.log_metric(model_name + " Error L"+str(lm), np.mean(all_errors))

    logger.log_metric(model_name + " Val Error All", np.mean(all_errors))
