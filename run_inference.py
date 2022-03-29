# from __future__ import print_function, absolute_import
import argparse
import enum
import os
import numpy as np
import sys
from comet_ml import Experiment

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything
# import kale.utils.logger as logging
from torchvision.transforms import Resize,InterpolationMode


from dataset import ASPIRELandmarks
from config import get_cfg_defaults
from model_trainer import UnetTrainer

from visualisation import visualize_predicted_heatmaps, visualize_image_multiscaleheats_pred_coords, visualize_heat_pred_coords
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
    # cfg.freeze()
    print(cfg)

    cfg.DATASET.DEBUG=False
    # cfg.INFERENCE.EVALUATION_MODE = "use_input_size"

    seed = cfg.SOLVER.SEED
    seed_everything(seed)



    writer = Experiment(
        api_key="B5Nk91E6iCmWvBznXXq3Ijhhp",
        project_name="landnnunet",
        workspace="schobs",
    )
        
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)  

    writer.set_name("test" + cfg.OUTPUT_DIR.split('/')[-1])
    writer.add_tag("test")




    
 #  Load Model
    model_paths = []
    # model_names = ["model_best_valid_loss", "model_ep_549", "model_best_valid_coord_error", "model_latest"] 
    model_names = ["model_ep_249", "model_best_valid_loss"] 

    for name in model_names:
        model_paths.append(os.path.join(cfg.OUTPUT_DIR, (name+ ".model")))

    for i in range(len(model_paths)):
        run_inference_model(writer, cfg, model_paths[i], model_names[i])





def run_inference_model(logger, cfg, model_path, model_name):


    test_dataset = ASPIRELandmarks(
        annotation_path =cfg.DATASET.SRC_TARGETS,
        landmarks = cfg.DATASET.LANDMARKS,
        split = "training",
        root_path = cfg.DATASET.ROOT,
        sigma = cfg.MODEL.GAUSS_SIGMA,
        cv = 1,
        # cache_data = cfg.TRAINER.CACHE_DATA,
        cache_data = True,
        # data_augmentation =self.model_config.DATASET.DATA_AUG,

        normalize=True,
        num_res_supervisions = cfg.SOLVER.NUM_RES_SUPERVISIONS,
        original_image_size= cfg.DATASET.ORIGINAL_IMAGE_SIZE,
        input_size =  cfg.DATASET.INPUT_SIZE,
        hm_lambda_scale = cfg.MODEL.HM_LAMBDA_SCALE,

 

    )

   


    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)


    

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
        print(i)
        # print("use full res coords, ",  use_full_res_coords, "resize first ", resize_first)

        if use_full_res_coords:
            targ_coords = data_dict['full_res_coords'].cpu().detach().numpy()
        else:
            # print("before: ", data_dict['target_coords'].shape,  data_dict['target_coords'])
            # targ_coords = np.flip(data_dict['target_coords'].cpu().detach().numpy(), axis=2)
            targ_coords = data_dict['target_coords'].cpu().detach().numpy()

            # targ_coords =np.array([[x[1], x[0]] for x in data_dict['target_coords'].cpu().detach().numpy()])

        # print("idx: ", data_dict['image_path'])
        # print("img: ", data_dict['image'])

        # print("TC: ", targ_coords)
        # print("FRC: ", data_dict['full_res_coords'].cpu().detach().numpy())

            
        #debugging for resizing the heatmap
        # final_heatmap = Resize(cfg.DATASET.ORIGINAL_IMAGE_SIZE, interpolation=  InterpolationMode.BICUBIC)(final_heatmap)

        pred_heatmaps, final_pred_heatmap, pred_coords, loss = trainer.predict_heatmaps_and_coordinates(data_dict, return_all_layers=True, resize_to_og=resize_first)    
        pred_coords = pred_coords.cpu().detach().numpy()
        # pred_heatmaps = [x.cpu().detach().numpy() ]


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
            if cfg.INFERENCE.EVALUATION_MODE == 'scale_heatmap_first' :

                original_image = np.asarray(test_dataloader.dataset.datatype_load(data_dict["image_path"][0]))
                print("target Coords %s, predicted coords %s and error: %s. Loss: %s " % (targ_coords, pred_coords, coord_error, np.mean(loss)))

                
                print("ogi", original_image.shape, "mean of og image@ ", np.mean(original_image))
                visualize_heat_pred_coords(original_image, pred_coords[0], targ_coords[0])
                # visualize_image_multiscaleheats_pred_coords(original_image, pred_heatmaps[-1].cpu().detach().numpy()[0], final_pred_heatmap.cpu().detach().numpy()[0], pred_coords[0], targ_coords[0])
            elif cfg.INFERENCE.EVALUATION_MODE == 'use_input_size':
                original_image = data_dict['image'][0][0]
                print("target Coords %s, predicted coords %s and error: %s. Loss: %s " % (targ_coords, pred_coords, coord_error, np.mean(loss)))

                # print("ogi", original_image)
                visualize_image_multiscaleheats_pred_coords(original_image, pred_heatmaps[-1].cpu().detach().numpy()[0], final_pred_heatmap.cpu().detach().numpy()[0], pred_coords[0], targ_coords[0])
            else:
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


if __name__ == "__main__":
    main()
