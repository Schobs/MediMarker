"""
Centralised script for running workloads using the LaNNU-Net framework.

Authors: Lawrence Schobs, Ethan Jones
Last updated: 2023-01-05
"""
from datetime import datetime
import os

from comet_ml import Experiment
from utils.comet_logging.logging_utils import save_comet_html
import torch
from pytorch_lightning.utilities.seed import seed_everything
from pandas import ExcelWriter

from config import get_cfg_defaults
from datasets.dataset_index import DATASET_INDEX
from trainer.model_trainer_index import MODEL_TRAINER_INDEX
from utils.setup.argument_utils import arg_parse

def main():
    """The main for this domain adaptation example, showing the workflow"""
    cfg = arg_parse()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = cfg.SOLVER.SEED
    seed_everything(seed)
    os.makedirs(cfg.OUTPUT.OUTPUT_DIR, exist_ok=True)
    time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    fold = str(cfg.TRAINER.FOLD)

    #--------- CometML set-up ---------#
    exp_name = cfg.OUTPUT.OUTPUT_DIR.split('/')[-1] +"_Fold"+fold+"_" + str(time)
    if cfg.OUTPUT.USE_COMETML_LOGGING:
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
    torch.cuda.empty_cache()
    dataset_class=  DATASET_INDEX[cfg.DATASET.DATASET_CLASS]
    trainer = MODEL_TRAINER_INDEX[cfg.MODEL.ARCHITECTURE]

    #--------- Training ---------#
    if not cfg.TRAINER.INFERENCE_ONLY:
        if writer is not None:
            writer.add_tag("training")
        trainer = trainer(trainer_config= cfg, is_train=True, 
                          dataset_class=dataset_class, output_folder=cfg.OUTPUT.OUTPUT_DIR,
                        comet_logger=writer)
        trainer.initialize(training_bool=True)
        trainer.train()
        if writer is not None:
            writer.add_tag("completed training")
    else:
        if writer is not None:
            writer.add_tag("inference only")

        trainer = trainer(trainer_config= cfg, is_train=False, dataset_class=dataset_class, 
                          output_folder=cfg.OUTPUT.OUTPUT_DIR, comet_logger=writer)
        trainer.initialize(training_bool=False)

    #--------- Testing ---------#
    local_sum_path = os.path.join(cfg.OUTPUT.OUTPUT_DIR, "summary_results_fold"+fold+".xlsx")
    indv_sample_sum_path = os.path.join(cfg.OUTPUT.OUTPUT_DIR, "ensemble_individual_results_fold"+fold+".xlsx")
    all_model_summaries = {}
    all_model_individuals = {}
    if cfg.INFERENCE.ENSEMBLE_INFERENCE:
        if writer is not None:
            writer.add_tag("ensemble_inference")
        all_summary_results, ind_results = trainer.run_inference_ensemble_models("testing", cfg.INFERENCE.ENSEMBLE_CHECKPOINTS,
                                                                                 cfg.INFERENCE.DEBUG)
        html_to_log = save_comet_html(all_summary_results, ind_results)
        writer.log_html(html_to_log)
        if cfg.OUTPUT.RESULTS_CSV_APPEND is not None:
            output_append = "_"+ str(cfg.OUTPUT.RESULTS_CSV_APPEND)
        else:
            output_append = ""
        print("Saving summary of results locally to: {local_sum_path}")
        with ExcelWriter(os.path.join(cfg.OUTPUT.OUTPUT_DIR, "ensemble_summary_results_fold"+fold+output_append+".xlsx")) as writer_:
            for n, df in (all_summary_results).items():
                df.to_excel(writer_, n)
        print("Saving individual sample results locally to: {indv_sample_sum_path}")
        with ExcelWriter(os.path.join(cfg.OUTPUT.OUTPUT_DIR, "ensemble_individual_results_fold"+fold+output_append+".xlsx")) as writer_:
            for n, df in (ind_results).items():
                df.to_excel(writer_, n)
        writer.add_tag("completed ensemble_inference")
        writer.add_tag("completed inference")
    else:
        if cfg.INFERENCE.TTA_ENSEMBLE_INFERENCE:
            if writer is not None:
                writer.add_tag("tta_inference")
                all_summary_results, ind_results = trainer.run_inference_tta("testing", cfg.INFERENCE.ENSEMBLE_CHECKPOINTS,
                                                                             cfg.INFERENCE.DEBUG)
                html_to_log = save_comet_html(all_summary_results, ind_results)
                writer.log_html(html_to_log)
                if cfg.OUTPUT.RESULTS_CSV_APPEND is not None:
                    output_append = "_"+ str(cfg.OUTPUT.RESULTS_CSV_APPEND)
                else:
                    output_append = ""
                print("Saving summary of results locally to: {local_sum_path}")
                with ExcelWriter(os.path.join(cfg.OUTPUT.OUTPUT_DIR, "tta_summary_results_fold"+fold+output_append+".xlsx")) as writer_:
                    for n, df in (all_summary_results).items():
                        df.to_excel(writer_, n)
                print("Saving individual sample results locally to: {indv_sample_sum_path}")
                with ExcelWriter(os.path.join(cfg.OUTPUT.OUTPUT_DIR, "tta_individual_results_fold"+fold+output_append+".xlsx")) as writer_:
                    for n, df in (ind_results).items():
                        df.to_excel(writer_, n)
                writer.add_tag("completed tta_inference")
                writer.add_tag("completed inference")
        elif cfg.MODEL.CHECKPOINT:
            print("Loading provided checkpoint" + str(cfg.MODEL.CHECKPOINT))
            model_name = cfg.MODEL.CHECKPOINT.split('/')[-1].split(".model")[0]
            print(str(model_name) + "model now loaded")
            trainer.load_checkpoint(cfg.MODEL.CHECKPOINT, training_bool=False)
            summary_results, ind_results = trainer.run_inference(split="testing", debug=cfg.INFERENCE.DEBUG)
            all_model_summaries[model_name] = summary_results
            all_model_individuals[model_name] = ind_results
        else:
            print("Loading all models from: " +cfg.OUTPUT.OUTPUT_DIR)
            model_paths = []
            model_names = []
            models_to_test = ["model_best_valid_loss", "model_best_valid_coord_error", "model_latest"]
            for fname in os.listdir(cfg.OUTPUT.OUTPUT_DIR):
                if ("fold"+fold in fname and ".model" in fname) and any(substring in fname for substring in models_to_test):
                    model_names.append(fname.split(".model")[0])
            for name in model_names:
                model_paths.append(os.path.join(cfg.OUTPUT.OUTPUT_DIR, (name+ ".model")))
            for i in range(len(model_paths)):
                print("loading:" + str({model_paths[i]}))
                trainer.load_checkpoint( model_paths[i], training_bool=False)
                summary_results, ind_results = trainer.run_inference(split="testing", debug=cfg.INFERENCE.DEBUG)
                all_model_summaries[model_names[i]] = summary_results
                all_model_individuals[model_names[i]] = ind_results

        #--------- Record results ---------#
        if writer is not None:
            html_to_log = save_comet_html(all_model_summaries, all_model_individuals)
            writer.log_html(html_to_log)
            print("Logged all results to CometML.")
        if cfg.OUTPUT.RESULTS_CSV_APPEND is not None:
            output_append = "_"+ str(cfg.OUTPUT.RESULTS_CSV_APPEND)
        else:
            output_append = ""
        print("Saving summary of results locally to: " +local_sum_path)
        with ExcelWriter(os.path.join(cfg.OUTPUT.OUTPUT_DIR, "ensemble_summary_results_fold"+fold+output_append+".xlsx")) as writer_:
            for n, df in (all_model_summaries).items():
                df.to_excel(writer_, n)
        print("Saving individual sample results locally to: " +indv_sample_sum_path)
        with ExcelWriter(os.path.join(cfg.OUTPUT.OUTPUT_DIR, "ensemble_individual_results_fold"+fold+output_append+".xlsx")) as writer_:
            for n, df in (all_model_individuals).items():
                df.to_excel(writer_, n)
        if writer is not None:
            writer.add_tag("completed inference")

if __name__ == "__main__":
    main()
