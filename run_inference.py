# from __future__ import print_function, absolute_import
from utils.setup.argument_utils import arg_parse
import enum
import os
import numpy as np
import sys
from comet_ml import Experiment
from pandas import ExcelWriter

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything
# import kale.utils.logger as logging
from torchvision.transforms import Resize,InterpolationMode
import pandas as pd

from dataset import ASPIRELandmarks
from config import get_cfg_defaults
from model_trainer import UnetTrainer
from evaluation.localization_evaluation import success_detection_rate, generate_summary_df
from utils.setup.argument_utils import get_evaluation_mode
from visualisation import visualize_predicted_heatmaps, visualize_image_multiscaleheats_pred_coords, visualize_heat_pred_coords
import csv
from utils.comet_logging.logging_utils import save_comet_html

from inference.fit_gaussian import fit_gauss




def main():
    """The main for this domain adaptation example, showing the workflow"""
    cfg = arg_parse()


    # ---- setup device ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("==> Using device " + device)

   
    # cfg.DATASET.DEBUG=False
    # cfg.INFERENCE.EVALUATION_MODE = "use_input_size"

    seed = cfg.SOLVER.SEED
    seed_everything(seed)



    writer = Experiment(
        api_key="B5Nk91E6iCmWvBznXXq3Ijhhp",
        project_name="landnnunet",
        workspace="schobs",
    )
        
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)  

    writer.set_name(cfg.OUTPUT_DIR.split('/')[-1])
    writer.add_tag("aprilV2")




    fold = str(cfg.TRAINER.FOLD)
    #Load Models that were saved.
    model_paths = []
    model_names = []
    for fname in os.listdir(cfg.OUTPUT_DIR):
            if "fold"+fold in fname and ".model" in fname:
                model_names.append(fname.split(".model")[0])

    for name in model_names:
        model_paths.append(os.path.join(cfg.OUTPUT_DIR, (name+ ".model")))


    all_model_summaries = {}
    all_model_individuals = {}

    for i in range(len(model_paths)):
        summary_results, ind_results = run_inference_model(writer, cfg, model_paths[i], model_names[i], "testing")
        
        all_model_summaries[model_names[i]] = summary_results
        all_model_individuals[model_names[i]] = ind_results


        #Print results and Log to CometML
        print("Results for Model: ", model_paths[i])
        print("All Mean Error %s +/- %s" % (summary_results.loc["Error Mean", "All"], summary_results.loc["Error Std","All"]), end=" - ")
        writer.log_metric(model_names[i].split("_fold")[0] + " Error All Mean", summary_results.loc["Error Mean", "All"])
        writer.log_metric(model_names[i].split("_fold")[0] + " Error All Std", summary_results.loc["Error Std", "All"])

        for k in summary_results.index.values:
                if "SDR" in k:
                    print(" %s : %s," % (k, summary_results.loc[k, "All"] ), end="")
                    writer.log_metric(model_names[i].split("_fold")[0] + " " + k,summary_results.loc[k, "All"])

        
        print("\n Individual Results: \n")
        for lm in range(len(cfg.DATASET.LANDMARKS)):
            print("Landmark %s Mean Error %s +/- %s " % (lm, summary_results.loc["Error Mean","L"+str(lm)], summary_results.loc["Error Std","L"+str(lm)] ), end=", ")
            for k in summary_results.index.values:
                if "SDR" in k:
                    print(" %s : %s," % (k, summary_results.loc[k,"L"+ str(lm)] ), end="")
            print("\n")


    #Now Save all model results to a spreadsheet

    html_to_log = save_comet_html(all_model_summaries, all_model_individuals)
    writer.log_html(html_to_log)
    print("Logged all results to CometML.")

    print("saving summary of results locally to: ", os.path.join(cfg.OUTPUT_DIR, "T_summary_results_fold"+fold +".xlsx"))
    with ExcelWriter(os.path.join(cfg.OUTPUT_DIR, "T_summary_results_fold"+fold +".xlsx")) as writer:
        for n, df in (all_model_summaries).items():
            df.to_excel(writer, n)

    print("saving individual sample results locally to: ", os.path.join(cfg.OUTPUT_DIR, "T_individual_results_fold"+fold +".xlsx"))
    with ExcelWriter(os.path.join(cfg.OUTPUT_DIR, "T_individual_results_fold"+fold +".xlsx")) as writer:
        for n, df in (all_model_individuals).items():
            df.to_excel(writer, n)













#  #  Load Model
#     model_paths = []
#     # model_names = ["model_best_valid_loss", "model_ep_549", "model_best_valid_coord_error", "model_latest"] 
#     # model_names = [ "model_best_valid_loss", "model_ep_949", "model_ep_249"] 
#     # model_names = ["model_best_valid_coord_error", "model_best_valid_loss", "model_ep_949", "model_ep_749"] 
#     model_names = ["model_best_valid_coord_error_fold" +fold  ] 
    
#     all_summmaries = {}
#     all_individuals = {}
#     for name in model_names:
#         model_paths.append(os.path.join(cfg.OUTPUT_DIR, (name+ ".model")))

#     for i in range(len(model_paths)):
#         summary_results, ind_results = run_inference_model(writer, cfg, model_paths[i], model_names[i], "testing")
#         print("Summary Results: \n ", summary_results)
#         all_summmaries[model_names[i]] = summary_results 
#         all_individuals[model_names[i]] = ind_results 

#     html_to_log = save_comet_html(all_summmaries, all_individuals)
#     writer.log_html(html_to_log)





def run_inference_model(logger, cfg, model_path, model_name, split):

    print("Running inference")
    test_dataset = ASPIRELandmarks(
        annotation_path =cfg.DATASET.SRC_TARGETS,
        landmarks = cfg.DATASET.LANDMARKS,
        split = split,
        root_path = cfg.DATASET.ROOT,
        sigma = cfg.MODEL.GAUSS_SIGMA,
        cv = cfg.TRAINER.FOLD,
        cache_data = cfg.TRAINER.CACHE_DATA,
        # cache_data = True,
        # data_augmentation =self.model_config.DATASET.DATA_AUG,

        normalize=True,
        num_res_supervisions = cfg.SOLVER.NUM_RES_SUPERVISIONS,
        original_image_size= cfg.DATASET.ORIGINAL_IMAGE_SIZE,
        input_size =  cfg.DATASET.INPUT_SIZE,
        hm_lambda_scale = cfg.MODEL.HM_LAMBDA_SCALE,
        regress_sigma = cfg.SOLVER.REGRESS_SIGMA


 

    )

   


    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)


    

    trainer = UnetTrainer(model_config= cfg, output_folder=cfg.OUTPUT_DIR)
    trainer.load_checkpoint(model_path, training_bool=False)
    
    all_errors = []
    landmark_errors = [[] for x in range(len(cfg.DATASET.LANDMARKS)) ]
    
    #get evaluation mode parameters
    use_full_res_coords, resize_first = get_evaluation_mode(cfg.INFERENCE.EVALUATION_MODE, cfg.DATASET.ORIGINAL_IMAGE_SIZE, cfg.DATASET.INPUT_SIZE)

    temp_results = []
    ind_results = []
    for i, data_dict in enumerate(test_dataloader):


        if use_full_res_coords:
            targ_coords = data_dict['full_res_coords'].cpu().detach().numpy()
        else:
          
            targ_coords = data_dict['target_coords'].cpu().detach().numpy()


        pred_heatmaps, final_pred_heatmap, pred_coords, loss = trainer.predict_heatmaps_and_coordinates(data_dict, return_all_layers=True, resize_to_og=resize_first)    
        pred_coords = pred_coords.cpu().detach().numpy()



        #If we don't resize the heatmap first and we want to evaluate on full image size, we need to scale coordinates up.
        if use_full_res_coords and not resize_first :
            downscale_factor = [cfg.DATASET.ORIGINAL_IMAGE_SIZE[0]/cfg.DATASET.INPUT_SIZE[0], cfg.DATASET.ORIGINAL_IMAGE_SIZE[1]/cfg.DATASET.INPUT_SIZE[1]]
            pred_coords = pred_coords * downscale_factor

        coord_error = np.linalg.norm((pred_coords- targ_coords), axis=2)

        # Fit the predicted heatmap into a Gaussian
        # print("final pred heatmap shape : ", final_pred_heatmap.shape, " pred_coords shape", pred_coords.shape)
        # for lm_ind in range(final_pred_heatmap.shape[1]):
        #     # if lm_ind == 4:
        #     fit_gauss(final_pred_heatmap[0][lm_ind].cpu().detach().numpy(), pred_coords[0][lm_ind], targ_coords = targ_coords[0][lm_ind], visualize=True)
         
        # exit()
        for idx_c, sample in enumerate(coord_error):

            temp_results.append({"uid": data_dict["uid"][idx_c], "ind_errors": coord_error[idx_c]})

            ind_result_sample = {"uid": data_dict["uid"][idx_c], "Error All Mean": np.mean(sample), "Error All Std ": np.std(sample)}
            for idx, er in enumerate(sample):
                ind_result_sample["L"+str(idx)] = er
                all_errors.append(er)
                landmark_errors[idx].append(er)

            ind_results.append(ind_result_sample)

        if cfg.DATASET.DEBUG:

            if cfg.INFERENCE.EVALUATION_MODE == 'scale_heatmap_first' :
                original_image = np.asarray(test_dataloader.dataset.datatype_load(data_dict["image_path"][0]))
                print("target Coords %s, predicted coords %s and error: %s. Loss: %s " % (targ_coords, pred_coords, coord_error, np.mean(loss)))            
                print("ogi", original_image.shape, "mean of og image@ ", np.mean(original_image))
                visualize_heat_pred_coords(original_image, pred_coords[0], targ_coords[0])
                
            elif cfg.INFERENCE.EVALUATION_MODE == 'use_input_size':
                original_image = data_dict['image'][0][0]
                print("target Coords %s, predicted coords %s and error: %s. Loss: %s " % (targ_coords, pred_coords, coord_error, np.mean(loss)))
                visualize_image_multiscaleheats_pred_coords(original_image, pred_heatmaps[-1].cpu().detach().numpy()[0], final_pred_heatmap.cpu().detach().numpy()[0], pred_coords[0], targ_coords[0])
            
            else:
                pred_heatmaps.append(final_pred_heatmap)
                if isinstance(pred_heatmaps, list):
                    pred_heatmaps = [x.cpu().detach().numpy() for x in pred_heatmaps]
                else:
                    pred_heatmaps = pred_heatmaps.cpu().detach().numpy()
                print("target Coords %s, predicted coords %s and error: %s. Loss: %s " % (targ_coords, pred_coords, coord_error, np.mean(loss)))
                visualize_predicted_heatmaps(pred_heatmaps, pred_coords, targ_coords)



    #### Caclulate evaluation metrics ####

    # Success Detection Rate i.e. % images within error thresholds
    radius_list = [20,25,30,40]
    outlier_results = {}
    for rad in radius_list:
        out_res_rad = success_detection_rate(temp_results, rad)
        outlier_results[rad] = (out_res_rad)    
    
    #Generate summary Results
    summary_results = generate_summary_df(landmark_errors, outlier_results )
    ind_results = pd.DataFrame(ind_results)
    

    return summary_results, ind_results

if __name__ == "__main__":
    main()
