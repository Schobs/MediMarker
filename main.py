# from __future__ import print_function, absolute_import
from comet_ml import Experiment

import os
import numpy as np
import sys
from datetime import datetime
import torch

from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.tensorboard import SummaryWriter

from config import get_cfg_defaults
# from trainer.model_trainer_unet import UnetTrainer
# from trainer.model_trainer_phdnet import PHDNetTrainer

from pandas import ExcelWriter
from utils.setup.argument_utils import arg_parse
import csv

from utils.comet_logging.logging_utils import save_comet_html


from datasets.dataset_index import DATASET_INDEX
from trainer.model_trainer_index import MODEL_TRAINER_INDEX


def main():
    
    """The main for this domain adaptation example, showing the workflow"""
    cfg = arg_parse()


    # ---- setup device ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("==> Using device " + device)



    seed = cfg.SOLVER.SEED
    seed_everything(seed)
    

    os.makedirs(cfg.OUTPUT.OUTPUT_DIR, exist_ok=True)  
    time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    fold = str(cfg.TRAINER.FOLD)

    exp_name = cfg.OUTPUT.OUTPUT_DIR.split('/')[-1] +"_Fold"+fold+"_" + str(time)


    #Set up Comet logging
    if cfg.OUTPUT.USE_COMETML_LOGGING:
        #Set up cometml writer
        writer = Experiment(
            api_key=cfg.OUTPUT.COMET_API_KEY,
            project_name=cfg.OUTPUT.COMET_PROJECT_NAME,
            workspace=cfg.OUTPUT.COMET_WORKSPACE,
        )
        writer.set_name(exp_name)
        writer.add_tag("fold" + str(cfg.TRAINER.FOLD))

        for tag_ in cfg.OUTPUT.COMET_TAGS:
            writer.add_tag(str(tag_))

        with open(os.path.join(cfg.OUTPUT.OUTPUT_DIR, 'comet_'+exp_name+'.txt'), 'w') as f:
            f.write('link to exp: ' + writer.url)
    else:
        writer = None




    

    #clear cuda cache
    torch.cuda.empty_cache()

    # exit()
    #This is Tensorboard stuff. Useful for profiler to optimize.
    # t_writer = SummaryWriter("/mnt/bess/home/acq19las/landmark_unet/LaNNU-Net/profiler/")
    # cfg.DATASET.DEBUG = True
    # prof = torch.profiler.profile(
    #     schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler("/mnt/bess/home/acq19las/landmark_unet/LaNNU-Net/profiler/"),
    #     record_shapes=True,
    #     with_stack=True)


    #get dataset class based on dataset, it defaults to datasets.dataset_generic

    dataset_class=  DATASET_INDEX[cfg.DATASET.DATASET_CLASS]

    trainer = MODEL_TRAINER_INDEX[cfg.MODEL.ARCHITECTURE]
    # #Get model trainer (u-net or phdnet)
    # if cfg.MODEL.ARCHITECTURE == "U-Net":
    #     trainer = UnetTrainer
    # elif cfg.MODEL.ARCHITECTURE == "PHD-Net":
    #     trainer = PHDNetTrainer
    # else:
    #     raise ValueError("trainer not recognised.")

    #Trainer 

    ############ Training ############
    if not cfg.TRAINER.INFERENCE_ONLY:
        if writer is not None:
            writer.add_tag("training")

        trainer = trainer(trainer_config= cfg, is_train=True, dataset_class=dataset_class, output_folder=cfg.OUTPUT.OUTPUT_DIR, comet_logger=writer)
        trainer.initialize(training_bool=True)
        trainer.train()

        if writer is not None:
            writer.add_tag("completed training")

    else:
        if writer is not None:
            writer.add_tag("inference only")

        trainer = trainer(trainer_config= cfg, is_train=False, dataset_class=dataset_class, output_folder=cfg.OUTPUT.OUTPUT_DIR, comet_logger=writer)
        trainer.initialize(training_bool=False)

    ########### Testing ##############
    print("\n Testing")

    all_model_summaries = {}
    all_model_individuals = {}

    #if ensemble inference, need to pass a list of checkpoint_paths to the model_trainer.
    # In the trainer, for each minibatch, I need to perform inference for each model and saving the model outputs in a list.
    # After the minibatch, I need to get S-MHA and a model prediction, E-MHA, E-CPV and the model predictions too.
    # Then I need to update a csv file per minibatch, NOT save to a list and write at the end.
    # Use the first model in the list as S-MHA.
    if cfg.INFERENCE.ENSEMBLE_INFERENCE:

        if writer is not None:
            writer.add_tag("ensemble_inference")

        # run_inference_ensemble_models(self, split, checkpoint_list, debug=False)
        all_summary_results, ind_results = trainer.run_inference_ensemble_models(split="testing", checkpoint_list=cfg.INFERENCE.ENSEMBLE_CHECKPOINTS, debug=cfg.INFERENCE.DEBUG)
        
        # ind_results = {"ensemble_results": ind_results}
        #Now Save all model results to a spreadsheet
        html_to_log = save_comet_html(all_summary_results, ind_results)
        writer.log_html(html_to_log)
        # print("Logged all results to CometML.")
        if cfg.OUTPUT.RESULTS_CSV_APPEND is not None:
            output_append = "_"+ str(cfg.OUTPUT.RESULTS_CSV_APPEND)
        else:
            output_append = ""

        print("saving summary of results locally to: ", os.path.join(cfg.OUTPUT.OUTPUT_DIR, "summary_results_fold"+fold +".xlsx"))
        with ExcelWriter(os.path.join(cfg.OUTPUT.OUTPUT_DIR, "ensemble_summary_results_fold"+fold+output_append +".xlsx")) as writer_:
            for n, df in (all_summary_results).items():
                df.to_excel(writer_, n)

        print("saving individual sample results locally to: ", os.path.join(cfg.OUTPUT.OUTPUT_DIR, "ensemble_individual_results_fold"+fold +".xlsx"))
        with ExcelWriter(os.path.join(cfg.OUTPUT.OUTPUT_DIR, "ensemble_individual_results_fold"+fold +output_append+".xlsx")) as writer_:
            for n, df in (ind_results).items():
                df.to_excel(writer_, n)
        writer.add_tag("completed ensemble_inference")
        writer.add_tag("completed inference")


    else:
        if cfg.MODEL.CHECKPOINT:
            print("loading provided checkpoint",cfg.MODEL.CHECKPOINT)

            model_name = cfg.MODEL.CHECKPOINT.split('/')[-1].split(".model")[0]
            print("model name: ", model_name)

            trainer.load_checkpoint(cfg.MODEL.CHECKPOINT, training_bool=False)
            summary_results, ind_results = trainer.run_inference(split="testing", debug=cfg.INFERENCE.DEBUG)

            all_model_summaries[model_name] = summary_results
            all_model_individuals[model_name] = ind_results

        else:
            print("Loading all models in cfg.OUTPUT.OUTPUT_DIR: ", cfg.OUTPUT.OUTPUT_DIR)
            #Load Models that were saved.
            model_paths = []
            model_names = []
            models_to_test =   ["model_best_valid_loss", "model_best_valid_coord_error", "model_latest"]

            for fname in os.listdir(cfg.OUTPUT.OUTPUT_DIR):
                if ("fold"+fold in fname and ".model" in fname) and any(substring in fname for substring in models_to_test):
                    model_names.append(fname.split(".model")[0])

            for name in model_names:
                model_paths.append(os.path.join(cfg.OUTPUT.OUTPUT_DIR, (name+ ".model")))



            
            # model_paths = ["/shared/tale2/Shared/schobs/landmark_unet/lannUnet_exps/ISBI/param_search/ISBI_512F_512Res_8GS_4MFR_AugAC_DS5/model_best_valid_coord_error_fold0.model"]

            
            for i in range(len(model_paths)):
                print("loading ", model_paths[i])
                trainer.load_checkpoint( model_paths[i], training_bool=False)
                # summary_results, ind_results = run_inference_model(writer, cfg, model_paths[i], model_names[i], "testing")
                summary_results, ind_results = trainer.run_inference(split="testing", debug=cfg.INFERENCE.DEBUG)

                # print(ind_results)
                # print(summary_results)

                all_model_summaries[model_names[i]] = summary_results
                all_model_individuals[model_names[i]] = ind_results


        #Now Save all model results to a spreadsheet
        if writer is not None:
            html_to_log = save_comet_html(all_model_summaries, all_model_individuals)
            writer.log_html(html_to_log)
            print("Logged all results to CometML.")

        if cfg.OUTPUT.RESULTS_CSV_APPEND is not None:
            output_append = "_"+ str(cfg.OUTPUT.RESULTS_CSV_APPEND)
        else:
            output_append = ""

        print("saving summary of results locally to: ", os.path.join(cfg.OUTPUT.OUTPUT_DIR, "summary_results_fold"+fold +".xlsx"))
        with ExcelWriter(os.path.join(cfg.OUTPUT.OUTPUT_DIR, "summary_results_fold"+fold+output_append +".xlsx")) as writer_:
            for n, df in (all_model_summaries).items():
                df.to_excel(writer_, n)

        print("saving individual sample results locally to: ", os.path.join(cfg.OUTPUT.OUTPUT_DIR, "individual_results_fold"+fold +".xlsx"))
        with ExcelWriter(os.path.join(cfg.OUTPUT.OUTPUT_DIR, "individual_results_fold"+fold +output_append+".xlsx")) as writer_:
            for n, df in (all_model_individuals).items():
                df.to_excel(writer_, n)

        if writer is not None:
            writer.add_tag("completed inference")


if __name__ == "__main__":
    main()
