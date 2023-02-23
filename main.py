# from __future__ import print_function, absolute_import
import os
from datetime import datetime
from comet_ml import Experiment
import torch
from pandas import ExcelWriter
from pytorch_lightning.utilities.seed import seed_everything

from datasets.dataset_index import DATASET_INDEX


from trainer.model_trainer_index import MODEL_TRAINER_INDEX
# from utils.logging.comet_logging import save_comet_html
from utils.setup.argument_utils import arg_parse

# from utils.logging.python_logger import get_logger, initialize_logging


def main():
    """The main file for training and testing the model. Everything is called from here."""

    # Read and infer arguments from yaml file
    cfg = arg_parse()

    # ---- setup device ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("==> Using device " + device)

    seed = cfg.SOLVER.SEED
    if seed is not None:
        seed_everything(seed)

    os.makedirs(cfg.OUTPUT.OUTPUT_DIR, exist_ok=True)
    time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    fold = str(cfg.TRAINER.FOLD)

    # ---- setup logger ----
    # initialize_logging()
    logger = get_logger(cfg.OUTPUT.LOGGER_OUTPUT)
    logger.info("Set logger output path: %s ", cfg.OUTPUT.LOGGER_OUTPUT)
    logger.info("Config \n %s ", cfg)

    exp_name = '_'.join(cfg.OUTPUT.OUTPUT_DIR.split(
        "/")[-2:]) + "_Fold" + fold + "_" + str(time)

    # Set up Comet logging
    if cfg.OUTPUT.USE_COMETML_LOGGING:
        # Set up cometml writer
        writer = Experiment(
            api_key=cfg.OUTPUT.COMET_API_KEY,
            project_name=cfg.OUTPUT.COMET_PROJECT_NAME,
            workspace=cfg.OUTPUT.COMET_WORKSPACE,
            display_summary_level=0
        )
        writer.set_name(exp_name)
        writer.add_tag("fold" + str(cfg.TRAINER.FOLD))

        for tag_ in cfg.OUTPUT.COMET_TAGS:
            writer.add_tag(str(tag_))

        logger.info("The comet.ml experiment HTML is %s ", writer.url)

    else:
        writer = None

    # clear cuda cache
    torch.cuda.empty_cache()

    # get dataset class based on dataset, it defaults to datasets.dataset_generic. Also get trainer
    dataset_class = DATASET_INDEX[cfg.DATASET.DATASET_CLASS]
    trainer = MODEL_TRAINER_INDEX[cfg.MODEL.ARCHITECTURE]

    ############ Set up model trainer, indicating if we are training or testing ############
    if not cfg.TRAINER.INFERENCE_ONLY:
        logger.info("TRAINING PHASE")

        if writer is not None:
            writer.add_tag("training")

        trainer = trainer(
            trainer_config=cfg,
            is_train=True,
            dataset_class=dataset_class,
            output_folder=cfg.OUTPUT.OUTPUT_DIR,
            comet_logger=writer,
        )
        trainer.initialize(training_bool=True)
        trainer.train()

        if writer is not None:
            writer.add_tag("completed training")
    else:
        if writer is not None:
            writer.add_tag("inference only")

        trainer = trainer(
            trainer_config=cfg,
            is_train=False,
            dataset_class=dataset_class,
            output_folder=cfg.OUTPUT.OUTPUT_DIR,
            comet_logger=writer,
        )
        trainer.initialize(training_bool=False)

    ########### TESTING ##############
    logger.info("TESTING PHASE")

    inference_split = cfg.INFERENCE.SPLIT
    all_model_summaries = {}
    all_model_individuals = {}

    # Ensemble inference
    if cfg.INFERENCE.ENSEMBLE_INFERENCE:

        if writer is not None:
            writer.add_tag("ensemble_inference")

        # run_inference_ensemble_models(self, split, checkpoint_list, debug=False)
        (
            all_model_summaries,
            all_model_individuals,
        ) = trainer.run_inference_ensemble_models(
            split=inference_split,
            checkpoint_list=cfg.INFERENCE.ENSEMBLE_CHECKPOINTS,
            debug=cfg.INFERENCE.DEBUG,
        )

        # Add ensemble marker to the output file
        if cfg.OUTPUT.RESULTS_CSV_APPEND is not None:
            output_append = "_ensemble_" + str(cfg.OUTPUT.RESULTS_CSV_APPEND)
        else:
            output_append = "_ensemble"

    else:
        writer.add_tag("single_inference")

        if cfg.MODEL.CHECKPOINT:
            logger.info("loading provided checkpoint %s", cfg.MODEL.CHECKPOINT)

            model_name = cfg.MODEL.CHECKPOINT.split("/")[-1].split(".model")[0]
            logger.info("model name: %s", model_name)

            trainer.load_checkpoint(cfg.MODEL.CHECKPOINT, training_bool=False)
            summary_results, ind_results = trainer.run_inference(
                split=inference_split, debug=cfg.INFERENCE.DEBUG
            )

            all_model_summaries[model_name] = summary_results
            all_model_individuals[model_name] = ind_results

        else:
            # Load top models predefined below.
            model_paths = []
            model_names = []
            models_to_test = [
                "model_best_valid_loss",
                "model_best_valid_coord_error",
                "model_latest",
            ]

            logger.info(
                "Loading MODELS that match substrings in: %s", models_to_test)

            for fname in os.listdir(cfg.OUTPUT.OUTPUT_DIR):
                if ("fold" + fold in fname and ".model" in fname) and any(
                    substring in fname for substring in models_to_test
                ):
                    model_names.append(fname.split(".model")[0])

            for name in model_names:
                model_paths.append(
                    os.path.join(cfg.OUTPUT.OUTPUT_DIR, (name + ".model"))
                )

            for i, model_p in enumerate(model_paths):
                logger.info("loading %s", model_p)
                trainer.load_checkpoint(model_p, training_bool=False)
                summary_results, ind_results = trainer.run_inference(
                    split=inference_split, debug=cfg.INFERENCE.DEBUG)

                all_model_summaries[model_names[i]] = summary_results
                all_model_individuals[model_names[i]] = ind_results

        # Add append to output file string
        if cfg.OUTPUT.RESULTS_CSV_APPEND is not None:
            output_append = "_" + str(cfg.OUTPUT.RESULTS_CSV_APPEND)
        else:
            output_append = ""

    ########### Now Save all model results to a spreadsheet #############
    if writer is not None:
        html_to_log = save_comet_html(
            all_model_summaries, all_model_individuals)
        writer.log_html(html_to_log)
        logger.info("Logged all results to CometML.")

    logger.info(
        "saving summary of results locally to: %s",
        os.path.join(cfg.OUTPUT.OUTPUT_DIR,
                     "summary_results_fold" + fold + ".xlsx"),
    )
    with ExcelWriter(  # pylint: disable=abstract-class-instantiated
        os.path.join(
            cfg.OUTPUT.OUTPUT_DIR,
            "summary_results_fold" + fold + output_append + ".xlsx",
        )
    ) as writer_:
        for n, df in (all_model_summaries).items():
            # Prevent sheet name from being too long, max is 31 characters.
            # Asumes most important stuff is at the end
            if len(n) > 31:
                n = n[-31:]
            df.to_excel(writer_, n)

    logger.info(
        "saving individual sample results locally to: %s",
        os.path.join(
            cfg.OUTPUT.OUTPUT_DIR,
            "individual_results_fold" + fold + output_append + ".xlsx",
        ),
    )
    with ExcelWriter(  # pylint: disable=abstract-class-instantiated
        os.path.join(
            cfg.OUTPUT.OUTPUT_DIR,
            "individual_results_fold" + fold + output_append + ".xlsx",
        )
    ) as writer_:
        for n, df in (all_model_individuals).items():
            # Prevent sheet name from being too long, max is 31 characters.
            # Asumes most important stuff is at the end
            if len(n) > 31:
                n = n[-31:]
            df.to_excel(writer_, n)

    if writer is not None:
        writer.add_tag("completed inference")
        logger.info("Experiment found:at %s" % writer.url)

    logger.info("Done!")


if __name__ == "__main__":
    main()
