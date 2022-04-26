# from __future__ import print_function, absolute_import
from comet_ml import Experiment

import os
import numpy as np
import sys

import torch

from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.tensorboard import SummaryWriter

from config import get_cfg_defaults
from model_trainer import UnetTrainer
from visualisation import visualize_predicted_heatmaps
from run_inference import run_inference_model
from pandas import ExcelWriter
from utils.setup.argument_utils import arg_parse
# from inference import run_inference_model
import csv

from utils.comet_logging.logging_utils import save_comet_html




def main():
    
    """The main for this domain adaptation example, showing the workflow"""
    cfg = arg_parse()
    print("Config: \n ", cfg)


    # ---- setup device ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("==> Using device " + device)



    seed = cfg.SOLVER.SEED
    seed_everything(seed)

    #Set up cometml writer
    writer = Experiment(
        api_key="B5Nk91E6iCmWvBznXXq3Ijhhp",
        project_name="landnnunet",
        workspace="schobs",
        )


    fold = str(cfg.TRAINER.FOLD)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)  
    writer.set_name(cfg.OUTPUT_DIR.split('/')[-1] +"_Fold"+fold)
    writer.add_tag("testing")
    writer.add_tag("fold" + str(cfg.TRAINER.FOLD))


    #clear cuda cache
    torch.cuda.empty_cache()


    #This is Tensorboard stuff. Useful for profiler to optimize.
    # t_writer = SummaryWriter("/mnt/bess/home/acq19las/landmark_unet/LaNNU-Net/profiler/")
    # cfg.DATASET.DEBUG = True
    # prof = torch.profiler.profile(
    #     schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler("/mnt/bess/home/acq19las/landmark_unet/LaNNU-Net/profiler/"),
    #     record_shapes=True,
    #     with_stack=True)


    #Trainer 
    trainer = UnetTrainer(model_config= cfg, output_folder=cfg.OUTPUT_DIR, logger=writer)
    trainer.initialize(training_bool=True)

    trainer.train()


    ########### Testing ##############

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

    print("saving summary of results locally to: ", os.path.join(cfg.OUTPUT_DIR, "summary_results_fold"+fold +".xlsx"))
    with ExcelWriter(os.path.join(cfg.OUTPUT_DIR, "summary_results_fold"+fold +".xlsx")) as writer:
        for n, df in (all_model_summaries).items():
            df.to_excel(writer, n)

    print("saving individual sample results locally to: ", os.path.join(cfg.OUTPUT_DIR, "individual_results_fold"+fold +".xlsx"))
    with ExcelWriter(os.path.join(cfg.OUTPUT_DIR, "individual_results_fold"+fold +".xlsx")) as writer:
        for n, df in (all_model_individuals).items():
            df.to_excel(writer, n)



if __name__ == "__main__":
    main()
