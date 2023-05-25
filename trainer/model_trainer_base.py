import copy
import os
import types
from inference.ensemble_inference_helper import EnsembleUncertainties
import torch
import numpy as np
from time import time
from utils.im_utils.visualisation import plot_target
from utils.logging.comet_logging import comet_log_yaml_parameters
from utils.model_utils.torch_to_onnx import torch_to_onnx
# import multiprocessing as mp
from utils.logging.dict_logger import DictLogger
from torch.cuda.amp import GradScaler, autocast
from evaluation.localization_evaluation import (
    success_detection_rate,
    generate_summary_df,
)
from torch.utils.data import DataLoader

from abc import ABC, abstractmethod
import imgaug
import pandas as pd

import logging
from utils.setup.argument_utils import checkpoint_loading_checking
from utils.setup.download_models import download_model_from_google_drive


class NetworkTrainer(ABC):
    """Super class for trainers. I extend this for trainers for U-Net and PHD-Net. They share some functions.y"""

    @abstractmethod
    def __init__(
        self,
        trainer_config,
        is_train=True,
        dataset_class=None,
        output_folder=None,
        comet_logger=None,
        profiler=None,
    ):
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # This is the trainer config dict
        self.trainer_config = trainer_config
        self.model_type = self.trainer_config.MODEL.ARCHITECTURE
        self.is_train = is_train
        self.validation_log_heatmaps = trainer_config.TRAINER.VALIDATION_LOG_HEATMAPS
        self.inference_log_heatmaps = trainer_config.INFERENCE.LOG_HEATMAPS
        self.inference_log_heatmap_wo_noise = self.trainer_config.INFERENCE.LOG_HEATMAPS_WO_NOISE
        self.log_targ_hm = self.trainer_config.INFERENCE.LOG_HEATMAP_PLOT_TARG
        self.inference_eval_mode = self.trainer_config.INFERENCE.EVALUATION_MODE
        self.inference_intermediate_outputs = self.trainer_config.INFERENCE.SAVE_INTERMEDIATE_OUTPUTS_ONLY

        # Dataset class to use
        self.dataset_class = dataset_class

        self.to_pytorch_tensor = self.trainer_config.DATASET.TO_PYTORCH_TENSOR
        self.dataset_name = self.trainer_config.DATASET.NAME

        # Dataloader info
        self.data_loader_batch_size_train = self.trainer_config.SOLVER.DATA_LOADER_BATCH_SIZE_TRAIN
        self.data_loader_batch_size_eval = self.trainer_config.SOLVER.DATA_LOADER_BATCH_SIZE_EVAL

        self.num_batches_per_epoch = self.trainer_config.SOLVER.MINI_BATCH_SIZE
        self.gen_hms_in_mainthread = self.trainer_config.INFERRED_ARGS.GEN_HM_IN_MAINTHREAD

        self.sampler_mode = self.trainer_config.SAMPLER.SAMPLE_MODE
        self.standardize_landmarks = self.trainer_config.DATASET.STANDARDIZE_LANDMARKS
        # Patch centering args

        patch_sampler_generic_args = {"sample_patch_size": self.trainer_config.SAMPLER.PATCH.SAMPLE_PATCH_SIZE,
                                      "sample_patch_from_resolution": self.trainer_config.SAMPLER.PATCH.RESOLUTION_TO_SAMPLE_FROM}

        patch_sampler_bias_args = {"sampling_bias": self.trainer_config.SAMPLER.PATCH.SAMPLER_BIAS}

        patch_sampler_centring_train_args = {"xlsx_path": self.trainer_config.SAMPLER.PATCH.CENTRED_PATCH_COORDINATE_PATH,
                                             "xlsx_sheet": self.trainer_config.SAMPLER.PATCH.CENTRED_PATCH_COORDINATE_PATH_SHEET,
                                             "center_patch_jitter": self.trainer_config.SAMPLER.PATCH.CENTRED_PATCH_JITTER,
                                             "deterministic": self.trainer_config.SAMPLER.PATCH.CENTRED_PATCH_DETERMINISTIC,
                                             "safe_padding": 5
                                             }

        patch_sampler_centring_eval_args = {"xlsx_path": self.trainer_config.SAMPLER.PATCH.CENTRED_PATCH_COORDINATE_PATH,
                                            "xlsx_sheet": self.trainer_config.SAMPLER.PATCH.CENTRED_PATCH_COORDINATE_PATH_SHEET,
                                            "center_patch_jitter": self.trainer_config.SAMPLER.PATCH.CENTRED_PATCH_JITTER,
                                            "deterministic": self.trainer_config.SAMPLER.PATCH.CENTRED_PATCH_DETERMINISTIC_EVAL,
                                            "safe_padding": 5}

        self.train_dataset_patch_sampling_args = {"generic": patch_sampler_generic_args,
                                                  "biased": patch_sampler_bias_args,
                                                  "centred": patch_sampler_centring_train_args}

        self.eval_dataset_patch_sampling_args = {"generic": patch_sampler_generic_args,
                                                 "biased": patch_sampler_bias_args,
                                                 "centred": patch_sampler_centring_eval_args}

        self.generic_dataset_args = {"landmarks": self.trainer_config.DATASET.LANDMARKS,
                                     "annotation_path": self.trainer_config.DATASET.SRC_TARGETS,
                                     "image_modality": self.trainer_config.DATASET.IMAGE_MODALITY,
                                     "root_path": self.trainer_config.DATASET.ROOT,
                                     "fold": self.trainer_config.TRAINER.FOLD,
                                     "dataset_split_size": self.trainer_config.DATASET.TRAINSET_SIZE,
                                     "standardize_landmarks": self.trainer_config.DATASET.STANDARDIZE_LANDMARKS}

        self.data_aug_args_training = {"data_augmentation_strategy": self.trainer_config.SAMPLER.DATA_AUG,
                                       "data_augmentation_package": self.trainer_config.SAMPLER.DATA_AUG_PACKAGE,
                                       "guarantee_lms_image": self.trainer_config.SAMPLER.DATA_AUG_GUARANTEE_LMS_IN_IMAGE
                                       }
        self.data_aug_args_evaluation = {"data_augmentation_strategy": None,
                                         "data_augmentation_package": self.trainer_config.SAMPLER.DATA_AUG_PACKAGE,
                                         "guarantee_lms_image": self.trainer_config.SAMPLER.DATA_AUG_GUARANTEE_LMS_IN_IMAGE

                                         }
        self.label_generator_args = {
            "generate_heatmaps_here": not self.gen_hms_in_mainthread,
            "hm_lambda_scale": self.trainer_config.MODEL.HM_LAMBDA_SCALE
        }

        self.train_label_generator = self.eval_label_generator = None
        self.num_res_supervision = 1

        if self.sampler_mode in ["patch_bias", "patch_centred"]:
            self.training_resolution = self.trainer_config.SAMPLER.PATCH.RESOLUTION_TO_SAMPLE_FROM
        else:
            self.training_resolution = self.trainer_config.SAMPLER.INPUT_SIZE

        # Set up logger & profiler
        self.profiler = profiler
        self.comet_logger = comet_logger
        self.verbose_logging = self.trainer_config.OUTPUT.VERBOSE

        # Set up directories
        self.output_folder = output_folder
        self.local_logger_type = self.trainer_config.OUTPUT.LOCAL_LOGGER_TYPE
        self.google_drive_model_path = self.trainer_config.MODEL.MODEL_GDRIVE_DL_PATH

        # Trainer variables
        self.perform_validation = self.trainer_config.TRAINER.PERFORM_VALIDATION
        self.continue_checkpoint = self.trainer_config.MODEL.CHECKPOINT

        self.auto_mixed_precision = self.trainer_config.SOLVER.AUTO_MIXED_PRECISION

        self.max_num_epochs = self.trainer_config.SOLVER.MAX_EPOCHS
        # Regressing sigma parameters for heatmaps
        self.regress_sigma = self.trainer_config.SOLVER.REGRESS_SIGMA

        self.sigmas = [
            torch.tensor(x, dtype=float, device=self.device, requires_grad=True)
            for x in np.repeat(
                self.trainer_config.MODEL.GAUSS_SIGMA, len(self.generic_dataset_args["landmarks"])
            )
        ]

        # Validation parameters
        self.use_full_res_coords = self.trainer_config.INFERRED_ARGS.USE_FULL_RES_COORDS
        self.resize_first = self.trainer_config.INFERRED_ARGS.RESIZE_FIRST

        # Checkpointing params
        self.save_every = self.trainer_config.TRAINER.SAVE_EVERY
        self.validate_every = self.trainer_config.TRAINER.VALIDATE_EVERY

        self.save_latest_only = (
            self.trainer_config.TRAINER.SAVE_LATEST_ONLY
        )  # if false it will not store/overwrite _latest but separate files each

        # Loss function

        # To be initialised in the super class (here)
        self.was_initialized = False
        self.amp_grad_scaler = None
        self.train_dataloader = self.valid_dataloader = None
        self.dict_logger = None
        self.learnable_params = None

        self.logger = logging.getLogger()

        # Set up in init of extended class (child)
        self.network = types.SimpleNamespace()  # empty object
        self.optimizer = None
        self.loss = types.SimpleNamespace()  # empty object

        # Can be changed in extended class (child)
        self.early_stop_patience = self.trainer_config.SOLVER.EARLY_STOPPING_PATIENCE
        self.initial_lr = self.trainer_config.SOLVER.BASE_LR
        self.lr_policy = self.trainer_config.SOLVER.DECAY_POLICY

        # Inference params
        self.fit_gauss_inference = self.trainer_config.INFERENCE.FIT_GAUSS

        # Initialize
        self.epoch = 0
        self.best_valid_loss = 999999999999999999999999999
        self.best_valid_coord_error = 999999999999999999999999999
        self.best_valid_coords_epoch = 0
        self.best_valid_loss_epoch = 0
        self.epochs_wo_val_improv = 0
        self.print_initiaization_info = True
        self.epoch_start_time = time()
        self.epoch_end_time = time()

    def initialize(self, training_bool=True):
        """
        Initialize profiler, comet logger, training/val dataloaders, network, optimizer,
        loss, automixed precision
        """
        # torch.backends.cudnn.benchmark = True

        if self.profiler:
            self.logger.info("Initialized profiler")
            self.profiler.start()

        self.initialize_dataloader_settings()
        if training_bool:
            self.set_training_dataloaders()

        self.initialize_network()
        self.initialize_optimizer_and_scheduler()
        self.initialize_loss_function()
        self._maybe_init_amp()

        if self.comet_logger:
            # self.comet_logger.log_parameters(self.trainer_config)
            comet_log_yaml_parameters(self.comet_logger, self.trainer_config)

        # This is the logger that will save epoch results to log & log variables at inference, extend this for any extra
        # stuff you want to log/save at evaluation!
        self.dict_logger = DictLogger(
            len(self.generic_dataset_args["landmarks"]),
            self.regress_sigma,
            self.loss.loss_seperated_keys,
            self.dataset_class.additional_sample_attribute_keys,
            log_valid_heatmap=self.validation_log_heatmaps,
            log_inference_heatmap=self.inference_log_heatmaps,
            log_fitted_gauss=self.fit_gauss_inference,
            log_inference_heatmap_wo_like=self.inference_log_heatmap_wo_noise,
            model_type=self.local_logger_type
        )

        self.was_initialized = True

        self.maybe_download_model()
        self.maybe_load_checkpoint()

        self.print_initiaization_info = False

    @abstractmethod
    def initialize_network(self):
        """
        Initialize the network here!

        """

    @abstractmethod
    def initialize_optimizer_and_scheduler(self):
        """
        Initialize the optimizer and LR scheduler here!

        """

    @abstractmethod
    def initialize_loss_function(self):
        """
        Initialize the loss function here!

        """

    def maybe_update_lr(self, epoch=None, exponent=0.9):
        """
        if epoch is not None we overwrite epoch. Else we use epoch = self.epoch + 1

        (maybe_update_lr is called in on_epoch_end which is called before epoch is incremented.
        Therefore we need to do +1 here)

        """
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch
        poly_lr_update = self.initial_lr * (1 - ep / self.max_num_epochs) ** exponent

        self.optimizer.param_groups[0]["lr"] = poly_lr_update

    def _maybe_init_amp(self):
        """Initialize automatic mixed precision training if enabled."""
        if self.auto_mixed_precision and self.amp_grad_scaler is None:
            self.amp_grad_scaler = GradScaler()
            msg = "initialized auto mixed precision."
        else:
            msg = "Not initialized auto mixed precision."

        if self.print_initiaization_info:
            self.logger.info(msg)

    @abstractmethod
    def get_coords_from_heatmap(self, model_output, original_image_size):
        """
        Function to take model output and return coordinates & a Dict of any extra information to log (e.g. max of heatmap)
        """

    def train(self):
        """
        The main training loop. For every epoch we train and validate. Each training epoch covers a number of minibatches
        of a certain batch size, defined in the config file.
        """
        if not self.was_initialized:
            self.initialize(True)

        step = 0
        while self.epoch < self.max_num_epochs:

            self.epoch_start_time = time()

            self.network.train()

            generator = iter(self.train_dataloader)

            # We will log the training and validation info here. The keys we set describe all the info we are logging.
            per_epoch_logs = self.dict_logger.get_epoch_logger()

            self.logger.info("training, %s", self.epoch)
            # Train for X number of batches per epoch e.g. 250
            for _ in range(self.num_batches_per_epoch):
                l, generator = self.run_iteration(
                    generator,
                    self.train_dataloader,
                    backprop=True,
                    split="training",
                    log_coords=False,
                    logged_vars=per_epoch_logs,
                    restart_dataloader=True
                )
                if self.comet_logger:
                    self.comet_logger.log_metric("training loss iteration", l, step)
                step += 1

            if self.epoch % self.validate_every == 0:
                # del generator
                self.logger.info("validation, %s", self.epoch)

                with torch.no_grad():
                    self.network.eval()
                    generator = iter(self.valid_dataloader)

                    # If debugging, need to log some extra info than
                    if self.trainer_config.INFERENCE.DEBUG:
                        valid_logs = self.dict_logger.get_evaluation_logger()
                    else:
                        valid_logs = per_epoch_logs

                    while generator != None:
                        l, generator = self.run_iteration(
                            generator,
                            self.valid_dataloader,
                            backprop=False,
                            split="validation",
                            log_coords=True,
                            logged_vars=valid_logs,
                            restart_dataloader=False,
                            debug=self.trainer_config.INFERENCE.DEBUG

                        )

            self.epoch_end_time = time()

            continue_training = self.on_epoch_end(per_epoch_logs)

            if not continue_training:
                if self.profiler:
                    self.profiler.stop()
                break

            self.epoch += 1

        # Save the final weights
        if self.comet_logger:
            self.logger.info("Logging weights as histogram...")
            weights = []
            for name in self.network.named_parameters():
                if "weight" in name[0]:
                    weights.extend(name[1].detach().cpu().numpy().tolist())
            self.comet_logger.log_histogram_3d(weights, step=self.epoch)

    def run_iteration(
        self,
        generator,
        dataloader,
        backprop,
        split,
        log_coords,
        logged_vars=None,
        debug=False,
        direct_data_dict=None,
        restart_dataloader=True,
    ):
        """Runs a single iteration of a forward pass. It can perform back-propagation (training) or not (validation, testing), indicated w/ bool backprop.
            It can also retrieve coordindates from predicted heatmap and log variables using DictLogger, dictated by the keys in the logged_vars dict.
            It will generate a batch of samples from the generator, unless a direct_data_dict is provided, in which case it will use that instead.

        Args:
            generator (Iterable): An iterable that generates a batch of samples (use a Pytorch DataLoader as the iterable type)
            dataloader (Pytorch Dataloader): The python dataloader that can be reinitialized if the generator runs out of samples.
            backprop (bool): Whether to perform backpropagation or not.
            split (str): String for split  (training, validation or testing)
            log_coords (bool): Whether to extract and log coordinates from the predicted heatmap.
            logged_vars (Dict, optional): A Dictionary with keys to log, derived from a template in the class DictLogger. Defaults to None.
            debug (bool, optional): Whether to debug the function. Defaults to False.
            direct_data_dict (Dict, optional): If not None, will directly perform forward pass on this rather than iterate the generator. Defaults to None.

        Returns:
            loss: The loss value for the iteration
            generator: The generator, which is now one iteration ahead from the input generator, or may have been reinitialized if it ran out of samples (if training).
        """

        # We can either give the generator to be iterated or a data_dict directly
        if direct_data_dict is None:
            try:
                data_dict = next(generator)

            except StopIteration:
                if not restart_dataloader:
                    return 0, None
                else:
                    generator = iter(dataloader)
                    data_dict = next(generator)
        else:
            data_dict = direct_data_dict

        data = (data_dict["image"]).to(self.device)

        # torch_to_onnx(self.network, data, self.output_folder+"/model.onnx")

        # This happens when we regress sigma with > 0 workers due to multithreading issues.
        # Currently does not support patch-based, which is raised on run of programme by argument checker.
        if self.gen_hms_in_mainthread:
            data_dict["label"] = self.generate_heatmaps_batch(data_dict, dataloader)

        # Put targets to device
        target = {
            key: (
                [x.to(self.device) for x in val]
                if isinstance(val, list)
                else val.to(self.device)
            )
            for key, val in data_dict["label"].items()
        }

        self.optimizer.zero_grad()

        # Run the forward pass, using auto mixed precision if enabled
        if self.auto_mixed_precision:
            with autocast():
                output = self.network(data)
                del data
                # Only attempts loss if annotations avaliable for entire batch
                if all(data_dict["annotation_available"]):
                    l, loss_dict = self.loss(output, target, self.sigmas)
                    if backprop:
                        self.amp_grad_scaler.scale(l).backward()
                        self.amp_grad_scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.learnable_params, 12)
                        self.amp_grad_scaler.step(self.optimizer)
                        self.amp_grad_scaler.update()
                        if self.regress_sigma:
                            self.update_dataloader_sigmas(self.sigmas)

                else:
                    l = torch.tensor(0).to(self.device)
                    loss_dict = {}

        else:
            output = self.network(data)
            del data

            # Only attempts loss if annotations avaliable for entire batch
            if all(data_dict["annotation_available"]):
                l, loss_dict = self.loss(output, target, self.sigmas)

                if backprop:
                    l.backward()
                    torch.nn.utils.clip_grad_norm_(self.learnable_params, 12)
                    self.optimizer.step()
                    if self.regress_sigma:
                        self.update_dataloader_sigmas(self.sigmas)
            else:
                l = torch.tensor(0).to(self.device)
                loss_dict = {}

        # Log info from this iteration.
        if list(logged_vars.keys()) != []:
            with torch.no_grad():

                (
                    pred_coords,
                    pred_coords_input_size,
                    extra_info,
                    target_coords,
                ) = self.maybe_get_coords(output, log_coords, data_dict)

                logged_vars = self.dict_logger.log_key_variables(
                    logged_vars,
                    pred_coords,
                    extra_info,
                    target_coords,
                    loss_dict,
                    data_dict,
                    log_coords,
                    split,
                )
                if debug:
                    debug_vars = [
                        x
                        for x in logged_vars["individual_results"]
                        if x["uid"] in data_dict["uid"]
                    ]
                    self.eval_label_generator.debug_prediction(
                        data_dict,
                        output,
                        pred_coords,
                        pred_coords_input_size,
                        debug_vars,
                        extra_info,
                    )

        if self.profiler:
            self.profiler.step()

        del output
        del target
        return l.detach().cpu().numpy(), generator

    def maybe_get_coords(self, output, log_coords, data_dict):
        """
        From output gets coordinates and extra info for logging. If log_coords is false,
        returns None for all. It also decides whether to resize heatmap, rescale coords
        depending on config settings.

        Args:
            output (_type_): _description_
            log_coords (_type_): _description_
            data_dict (_type_): _description_

        Returns:
            _type_: _description_
        """

        if log_coords:
            pred_coords_input_size, extra_info = self.get_coords_from_heatmap(
                output, data_dict["original_image_size"]
            )
            extra_info["target_coords_input_size"] = data_dict["target_coords"]

            pred_coords, target_coords = self.maybe_rescale_coords(
                pred_coords_input_size, data_dict
            )
            if self.log_targ_hm:
                extra_info["final_heatmaps"] = plot_target(extra_info["final_heatmaps"],
                                                           data_dict["target_coords"])

        else:
            pred_coords = extra_info = target_coords = pred_coords_input_size = None

        return pred_coords, pred_coords_input_size, extra_info, target_coords

    def on_epoch_end(self, per_epoch_logs):
        """
         Always run to 1000 epochs
        :return:
        """

        new_best_valid = False
        new_best_coord_valid = False

        continue_training = self.epoch < self.max_num_epochs

        #######Logging some end of epoch info #############
        time_taken = self.epoch_end_time - self.epoch_start_time
        per_epoch_logs = self.dict_logger.log_epoch_end_variables(
            per_epoch_logs,
            time_taken,
            self.sigmas,
            self.optimizer.param_groups[0]["lr"],
        )

        # log them else they are lost!
        if self.comet_logger:
            self.dict_logger.log_dict_to_comet(
                self.comet_logger, per_epoch_logs, self.epoch
            )

        # Delete big data in logs for printing and memory
        per_epoch_logs.pop("individual_results", None)
        per_epoch_logs.pop("final_heatmaps", None)
        per_epoch_logs.pop("final_heatmaps_wo_like_noise", None)

        per_epoch_logs.pop("individual_results_extra_keys", None)

        if self.verbose_logging:
            self.logger.info("Epoch %s logs: %s", self.epoch, per_epoch_logs)

        # Checks for it this epoch was best in validation loss or validation coord error!
        if per_epoch_logs["validation_all_loss_all"] < self.best_valid_loss:
            self.best_valid_loss = per_epoch_logs["validation_all_loss_all"]
            self.best_valid_loss_epoch = self.epoch
            new_best_valid = True

        if per_epoch_logs["valid_coord_error_mean"] < self.best_valid_coord_error:
            self.best_valid_coord_error = per_epoch_logs["valid_coord_error_mean"]
            self.best_valid_coords_epoch = self.epoch
            new_best_coord_valid = True
            self.epochs_wo_val_improv = 0
        else:
            self.epochs_wo_val_improv += 1

        if self.epochs_wo_val_improv == self.early_stop_patience:
            continue_training = False
            self.logger.info(
                "EARLY STOPPING. Validation Coord Error did not reduce for %s epochs. ", self.early_stop_patience
            )

        self.maybe_save_checkpoint(new_best_valid, new_best_coord_valid)

        if self.lr_policy == "poly":
            self.maybe_update_lr(epoch=self.epoch)

        return continue_training

    def maybe_save_checkpoint(self, new_best_valid_bool, new_best_valid_coord_bool):
        """
        Saves a checkpoint every save_ever epochs.
        :return:
        """

        fold_str = str(self.generic_dataset_args["fold"])
        if (self.epoch % self.save_every == 0) or (self.epoch == self.max_num_epochs - 1):
            self.logger.info("saving scheduled checkpoint file...")
            if not self.save_latest_only:
                ckpt_save_pth = os.path.join(self.output_folder, "model_ep_" +
                                             str(self.epoch) + "_fold" + fold_str + ".model")
                self.save_checkpoint(ckpt_save_pth)
                self.logger.info("saved checkpoint at %s", ckpt_save_pth)
            self.save_checkpoint(
                os.path.join(
                    self.output_folder, "model_latest_fold" + (fold_str) + ".model"
                )
            )
            self.logger.info("saved latest checkpoint")
        if new_best_valid_bool:
            self.logger.info(
                "saving scheduled checkpoint file as it's new best on validation set..."
            )
            self.save_checkpoint(
                os.path.join(
                    self.output_folder,
                    "model_best_valid_loss_fold" + fold_str + ".model",
                )
            )

            self.logger.info("saved checkpoint at new best valid loss: %s", self.best_valid_loss)

        if new_best_valid_coord_bool:
            self.save_checkpoint(
                os.path.join(
                    self.output_folder,
                    "model_best_valid_coord_error_fold" + fold_str + ".model",
                )
            )

            self.logger.info("saved checkpoint at new best valid coord error: %s", self.best_valid_coord_error)

    def maybe_rescale_coords(self, pred_coords, data_dict):
        """Maybe rescale coordinates based on evaluation parameters, and decide which target coords to evaluate against.
            Cases C1:4:
            C1) used full-scale image to train or resized heatmap already: leave predicted, use full-res target
            C2) used low-scale image to train and rescaling coordinates up to full-scale image size
            C3) use low-scale image to train, want to eval on low-scale coordinates
        Args:
            pred_coords (tensor): coords extracted from output heatmap
            data_dict (dict): dataloader sample dictionary

        Returns:
            tensor, tensor: predicted coordinates and target coordinates for evaluation
        """

        # Don't worry in case annotations are not present since these are 0,0 anyway. this is handled elesewhere
        # C1
        if self.use_full_res_coords:
            target_coords = data_dict["full_res_coords"].to(self.device)  # C1
        else:
            target_coords = np.round(data_dict["target_coords"]).to(
                self.device
            )  # C3 (and C1 if input size == full res size so full & target the same)

        # C2
        if self.use_full_res_coords and not self.resize_first:

            # If we are using patch_centred sampling, we are basing prediction on a smaller patch
            # containing the landmark, so we need to add the patch corner to the prediction.
            if not torch.is_tensor(pred_coords):
                pred_coords = torch.tensor(pred_coords).to(self.device)
            if self.sampler_mode == "patch_centred":
                x_y_corner = torch.unsqueeze(torch.tensor(
                    [[data_dict["x_y_corner"][0][x], data_dict["x_y_corner"][1][x]] for x in range(len(data_dict["x_y_corner"][0]))]), axis=1).to(self.device)
                pred_coords = torch.add(pred_coords, x_y_corner).to(self.device)

            upscale_factor = torch.tensor(data_dict["resizing_factor"]).to(self.device)

            upscaled_coords = torch.mul(pred_coords, upscale_factor)
            pred_coords = torch.round(upscaled_coords).to(self.device)
            # pred_coords = pred_coords * upscale_factor

        return pred_coords, target_coords

    # @abstractmethod
    def patchify_and_predict(self, single_sample, logged_vars):
        """Function that takens in a large input image, patchifies it and runs each patch through the model & stitches heatmap together

        # 1) should split up into patches of given patch-size.
        # 2) should run patches through in batches using run_iteration, NOT LOGGING ANYTHING but needs to return the OUTPUTS somehow.
            MUST ADD OPTION TO RETURN OUTPUTS in run_iteration?
        # 3) Need to use method to stitch patches together (future, phdnet will use patch size 512 512 for now).
        # 4) call log_key_variables function now with the final big heatmap as the "output". The logging should work as usual from that.

        Returns:
            _type_: _description_
        """

    def get_intermediate_representations(self, dataloader, split, debug=False):

        # instantiate interested variables log!
        intermediate_outputs_dict = {}

        # network evaluation mode
        self.network.eval()

        # then iterate through dataloader and save to log
        all_layers_to_iterate = ["B", "D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7"]
        # all_layers_to_iterate = ["D6", "D7"]

        for layer in all_layers_to_iterate:
            # save_folder_int = self.output_folder+"/intermediate_outputs/fold" + \
            #     self.generic_dataset_args["fold"] + "/"+layer + "/"
            save_folder_int_split_by_split = self.output_folder+"/intermediate_outputs/fold" + \
                str(self.generic_dataset_args["fold"]) + "/"+layer + "/" + split + "/"

            # os.makedirs(save_folder_int, exist_ok=True)
            os.makedirs(save_folder_int_split_by_split, exist_ok=True)

            # intermediate_outputs_dict = {}
            generator = iter(dataloader)

            for c_ix, data_dict in enumerate(generator):
                self.logger.info("processing batch %s", c_ix)
                data = (data_dict["image"]).to(self.device)

                int_output = self.network.get_intermediate_representation(data, layer).cpu().detach().numpy()
                for idx, uid in enumerate(data_dict["uid"]):
                    # torch.save(int_output[idx], os.path.join(save_folder_int, uid+".pth"))
                    torch.save(int_output[idx], os.path.join(save_folder_int_split_by_split, uid+".pth"))

                    # intermediate_outputs_dict[uid]=int_output[idx]
                del int_output

            # torch.save(intermediate_outputs_dict, os.path.join(self.output_folder, "intermediate_outputs_dict_foldX_Y.pth"))
            # torch.save(intermediate_outputs_dict, os.path.join(self.output_folder,
            #                                                    "intermediate_outputs_dict_fold{}_{}_{}.pth".format(self.generic_dataset_args["fold"], split, layer)))
    # @abstractmethod

    def run_inference(self, split, debug=False):
        """Function to run inference on a full sized input

        # 0) instantitate test dataset and dataloader
        # 1A) if FULL:
            i) iterate dataloader and run_iteration each time to go through and save results.
            ii) should run using run_iteration with logged_vars to log
        1B) if PATCHIFYING full_res_output  <- this can be fututre addition
            i) use patchify_and_predict to stitch hm together with logged_vars to log

        # 2) need a way to deal with the key dictionary & combine all samples
        # 3) need to put evaluation methods in evluation function & import and ues key_dict for analysis
        # 4) return individual results & do summary results.
        """

        # If trained using patch, return the full image, else ("full") will return the image size network was trained on.
        if self.sampler_mode in ["patch_bias", "patch_centred"]:
            if (
                self.trainer_config.SAMPLER.PATCH.INFERENCE_MODE
                == "patchify_and_stitch"
            ):
                # In this case we are patchifying the image
                inference_full_image = False
            else:
                # else we are doing it fully_convolutional
                inference_full_image = True
        else:
            # This case is the full sampler
            inference_full_image = True
        inference_resolution = self.training_resolution
        # Load dataloader (Returning coords dont matter, since that's handled in log_key_variables)
        test_dataset = self.get_evaluation_dataset(split, inference_resolution)
        test_batch_size = self.maybe_alter_batch_size(test_dataset, self.data_loader_batch_size_eval)

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=self.num_workers_cfg,
            persistent_workers=self.persist_workers,
            worker_init_fn=NetworkTrainer.worker_init_fn,
            pin_memory=True,
        )

        # instantiate interested variables log!
        evaluation_logs = self.dict_logger.get_evaluation_logger()

        # network evaluation mode
        self.network.eval()

        # then iterate through dataloader and save to log
        generator = iter(test_dataloader)
        if inference_full_image:
            if self.inference_intermediate_outputs:
                ind_results = self.get_intermediate_representations(test_dataloader, split, debug=debug)
                return None, ind_results

            else:
                while generator != None:
                    print("-", end="")
                    l, generator = self.run_iteration(
                        generator,
                        test_dataloader,
                        backprop=False,
                        split=split,
                        log_coords=True,
                        logged_vars=evaluation_logs,
                        debug=debug,
                        restart_dataloader=False
                    )

                    if self.inference_log_heatmaps:
                        self.save_heatmaps(evaluation_logs)

                        # per_epoch_logs.pop("individual_results", None)

                    for idx, results_dict in enumerate(evaluation_logs['individual_results']):
                        results_dict.pop("final_heatmaps", None)
                        results_dict.pop("final_heatmaps_wo_like_noise", None)

                del generator
                print()
        else:
            # this is where we patchify and stitch the input image
            raise NotImplementedError()

        summary_results, ind_results = self.evaluation_metrics(
            evaluation_logs["individual_results"], evaluation_logs["landmark_errors"]
        )

        return summary_results, ind_results

    def evaluation_metrics(self, individual_results, landmark_errors):
        """Function to calculate evaluation metrics."""
        radius_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50, 100]
        outlier_results = {}
        for rad in radius_list:
            out_res_rad = success_detection_rate(individual_results, rad)
            outlier_results[rad] = out_res_rad

        # Generate summary Results
        summary_results = generate_summary_df(landmark_errors, outlier_results)
        ind_results = pd.DataFrame(individual_results)
        ind_results = ind_results.drop(columns="final_heatmaps", errors='ignore')
        ind_results = ind_results.drop(columns="final_heatmaps_wo_like_noise", errors='ignore')

        return summary_results, ind_results

    def save_checkpoint(self, path):
        """Save model checkpoint to path

        Args:
            path (str): Path to save model checkpoint to.
        """
        state = {
            "epoch": self.epoch + 1,
            "state_dict": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_valid_loss": self.best_valid_loss,
            "best_valid_coord_error": self.best_valid_coord_error,
            "best_valid_loss_epoch": self.best_valid_loss_epoch,
            "best_valid_coords_epoch": self.best_valid_coords_epoch,
            "epochs_wo_improvement": self.epochs_wo_val_improv,
            "sigmas": self.sigmas,
            "training_sampler": self.sampler_mode,
            "training_resolution": self.training_resolution,
        }

        if self.amp_grad_scaler is not None:
            state["amp_grad_scaler"] = self.amp_grad_scaler.state_dict()

        torch.save(state, path)

    def run_inference_ensemble_models(self, split, checkpoint_list, debug=False):
        """Run inference on a list of checkpoints (ensemble) and return results and uncertainty measures.

        Args:
            split (str): Split to run inference on
            checkpoint_list ([str"]): List of paths of checkpoints to load
            debug (bool, optional): Debug mode. Defaults to False.

        Raises:
            NotImplementedError: Patch-based ensemble inference not implemented.

        Returns:
            all_summary_results: Summary of results for all uncertainty measures from EnsembleUncertainties
            ind_results: Individual results for each sample for all uncertainty measures from EnsembleUncertainties
        """

        # If trained using patch, return the full image, else ("full") will return the image size network was trained on.
        if self.sampler_mode in ["patch_bias", "patch_centred"]:
            if (
                self.trainer_config.SAMPLER.PATCH.INFERENCE_MODE
                == "patchify_and_stitch"
            ):
                # In this case we are patchifying the image
                inference_full_image = False
            else:
                # else we are doing it fully_convolutional
                inference_full_image = True
        else:
            # This case is the full sampler
            inference_full_image = True
        inference_resolution = self.training_resolution
        # Load dataloader (Returning coords dont matter, since that's handled in log_key_variables)
        test_dataset = self.get_evaluation_dataset(split, inference_resolution)
        test_batch_size = self.maybe_alter_batch_size(test_dataset, self.data_loader_batch_size_eval)

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=0,
            worker_init_fn=NetworkTrainer.worker_init_fn,
        )

        # Initialise ensemble postprocessing and uncertainty estimation
        uncertainty_estimation_keys = (
            self.trainer_config.INFERENCE.ENSEMBLE_UNCERTAINTY_KEYS
        )
        smha_model_idx = self.trainer_config.INFERENCE.UNCERTAINTY_SMHA_MODEL_IDX

        ensemble_handler = EnsembleUncertainties(
            uncertainty_estimation_keys, smha_model_idx, self.generic_dataset_args["landmarks"]
        )

        # network evaluation mode
        self.network.eval()

        # Initialise esenmble results dictionaries
        ensemble_result_dicts = {
            uncert_key: [] for uncert_key in ensemble_handler.uncertainty_keys
        }
        all_ind_errors = {
            uncert_key: [[] for x in range(len(self.generic_dataset_args["landmarks"]))]
            for uncert_key in ensemble_handler.uncertainty_keys
        }

        # Iterate through dataloader and save to log
        generator = iter(test_dataloader)
        if inference_full_image:
            # Need to load and get results from each checkpoint. Load checkpoint for each batch because of memory issues running through entire dataloader
            # and saving multiple outputs for every checkpoint. In future can improve this by going through X (e.g.200 samples/10 batches) before changing checkpoint.
            while generator != None:
                try:
                    evaluation_logs = self.dict_logger.ensemble_inference_log_template()
                    direct_data_dict = next(generator)
                    for ckpt in checkpoint_list:
                        self.load_checkpoint(ckpt, training_bool=False)
                        # Directly pass the next data_dict to run_iteration rather than iterate it within.
                        l, _ = self.run_iteration(
                            generator,
                            test_dataloader,
                            backprop=False,
                            split=split,
                            log_coords=True,
                            logged_vars=evaluation_logs,
                            debug=debug,
                            direct_data_dict=direct_data_dict,
                            restart_dataloader=False

                        )

                    # Analyse batch for s-mha, e-mha, and e-cpv and maybe errors (if we have annotations)
                    (
                        ensembles_analyzed,
                        ind_landmark_errors,
                    ) = ensemble_handler.ensemble_inference_with_uncertainties(
                        evaluation_logs
                    )

                    # Update the dictionaries with the results
                    for k_ in list(ensemble_result_dicts.keys()):
                        ensemble_result_dicts[k_].extend(ensembles_analyzed[k_])

                    for ens_key, coord_extact_methods in ind_landmark_errors.items():
                        for ile_idx, ind_lm_ers in enumerate(coord_extact_methods):
                            all_ind_errors[ens_key][ile_idx].extend(ind_lm_ers)

                except StopIteration:
                    generator = None
                print("-", end="")
            print("No more in generator")
            del generator

        else:
            # this is where we patchify and stitch the input image
            raise NotImplementedError()

        ind_results = {}
        all_summary_results = {}

        for u_key in uncertainty_estimation_keys:
            summary_results, ind_results_this = self.evaluation_metrics(
                ensemble_result_dicts[u_key], all_ind_errors[u_key]
            )
            ind_results[u_key] = ind_results_this
            all_summary_results[u_key] = summary_results

        return all_summary_results, ind_results

    def maybe_download_model(self):
        if self.google_drive_model_path:
            download_model_from_google_drive(self.google_drive_model_path,
                                             self.continue_checkpoint)

    def maybe_load_checkpoint(self):
        """Helper function from initialisation that loads checkpoint"""
        if self.continue_checkpoint:
            self.load_checkpoint(self.continue_checkpoint, self.is_train)

    def update_dataloader_sigmas(self, new_sigmas):
        """Update the dataset sigmas, used if regressing sigmas"""
        np_sigmas = [x.cpu().detach().numpy() for x in new_sigmas]
        self.train_dataloader.dataset.sigmas = np_sigmas
        self.valid_dataloader.dataset.sigmas = np_sigmas

    def load_checkpoint(self, model_path, training_bool):
        """Load checkpoint from path.

        Args:
            model_path (str): path to checjpoint
            training_bool (bool): If training or not.
        """
        if not self.was_initialized:
            self.initialize(training_bool)

        checkpoint_info = torch.load(model_path, map_location=self.device)
        self.epoch = checkpoint_info["epoch"]
        self.network.load_state_dict(checkpoint_info["state_dict"])
        self.optimizer.load_state_dict(checkpoint_info["optimizer"])

        if training_bool:
            self.best_valid_loss = checkpoint_info["best_valid_loss"]
            self.best_valid_loss_epoch = checkpoint_info["best_valid_loss_epoch"]
            self.best_valid_coord_error = checkpoint_info["best_valid_coord_error"]
            self.best_valid_coords_epoch = checkpoint_info["best_valid_coords_epoch"]
            self.epochs_wo_val_improv = checkpoint_info["epochs_wo_improvement"]

        # Allow legacy models to be loaded (they didn't use to save sigmas)
        if "sigmas" in checkpoint_info:
            self.sigmas = checkpoint_info["sigmas"]

        # if not saved, default to full since this was the only option for legacy models
        if "training_sampler" in checkpoint_info:
            self.sampler_mode = checkpoint_info["training_sampler"]
        else:
            self.sampler_mode = "full"

        # if not saved, default to input_size since this was the only option for legacy models
        if "training_resolution" in checkpoint_info:
            self.training_resolution = checkpoint_info["training_resolution"]
        else:
            self.training_resolution = self.trainer_config.SAMPLER.INPUT_SIZE

        checkpoint_loading_checking(self.trainer_config, self.sampler_mode, self.training_resolution)

        if self.auto_mixed_precision:
            self._maybe_init_amp()

            if "amp_grad_scaler" in checkpoint_info.keys():
                self.amp_grad_scaler.load_state_dict(checkpoint_info["amp_grad_scaler"])

        if self.print_initiaization_info:
            self.logger.info("Loaded checkpoint %s. Epoch: %s, ", model_path, self.epoch)

    def initialize_dataloader_settings(self):
        """Initializes dataloader settings. If debug use only main thread to load data bc we only
        want to show a single plot on screen.
        If num_workers=0 we are only using the main thread, so persist_workers = False.
        """

        if (
            self.trainer_config.SAMPLER.DEBUG
            or self.trainer_config.SAMPLER.NUM_WORKERS == 0
        ):
            self.persist_workers = False
            self.num_workers_cfg = 0
        else:
            self.persist_workers = True
            self.num_workers_cfg = self.trainer_config.SAMPLER.NUM_WORKERS

    def set_training_dataloaders(self):
        """
        set train_dataset, valid_dataset and train_dataloader and valid_dataloader here.
        """

        np_sigmas = [x.cpu().detach().numpy() for x in self.sigmas]

        #### Training dataset ####
        train_dataset = self.dataset_class(
            LabelGenerator=self.train_label_generator,
            split="training",
            sample_mode=self.sampler_mode,
            patch_sampler_args=self.train_dataset_patch_sampling_args,
            dataset_args=self.generic_dataset_args,
            data_aug_args=self.data_aug_args_training,
            label_generator_args=self.label_generator_args,
            sigmas=np_sigmas,
            cache_data=self.trainer_config.TRAINER.CACHE_DATA,
            num_res_supervisions=self.num_res_supervision,
            debug=self.trainer_config.SAMPLER.DEBUG,
            input_size=self.trainer_config.SAMPLER.INPUT_SIZE,
            to_pytorch=self.to_pytorch_tensor
        )

        #### Validation dataset ####

        # If not performing validation, just use the training set for validation.
        if self.perform_validation:
            validation_split = "validation"
        else:
            validation_split = "training"
            self.logger.warning(
                'WARNING: NOT performing validation. Instead performing "validation" on training set for coord error metrics.')

        # Image loading size different for patch vs. full image sampling
        if self.sampler_mode in ["patch_bias", "patch_centred"]:
            img_resolution = self.trainer_config.SAMPLER.PATCH.RESOLUTION_TO_SAMPLE_FROM
        else:
            img_resolution = self.trainer_config.SAMPLER.INPUT_SIZE

        valid_dataset = self.get_evaluation_dataset(validation_split, img_resolution)

        #### Create DataLoaders ####

        self.logger.info("Using %s Dataloader workers and persist workers bool : %s ",
                         self.num_workers_cfg, self.persist_workers)

        train_batch_size = self.maybe_alter_batch_size(train_dataset, self.data_loader_batch_size_train)
        valid_batch_size = self.maybe_alter_batch_size(valid_dataset, self.data_loader_batch_size_eval)

        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=self.num_workers_cfg,
            persistent_workers=self.persist_workers,
            worker_init_fn=NetworkTrainer.worker_init_fn,
            pin_memory=True,
        )
        self.valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=valid_batch_size,
            shuffle=True,
            num_workers=self.num_workers_cfg,
            persistent_workers=self.persist_workers,
            worker_init_fn=NetworkTrainer.worker_init_fn,
            pin_memory=True,
        )

    def get_evaluation_dataset(self, split, load_im_size):
        """Gets an evaluation dataset based on split given (must be "validation" or "testing").
            We do not perform patch sampling on evaluation dataset, always returning the full image (sample_mode = "full").
            Patchifying the evaluation image is too large memory constraint to do in batches here.

        Args:
            split (string): Which split of data to return ( "validation" or "testing")

        Returns:
            dataset: Dataset object
        """

        # assert split in ["validation", "testing"]
        np_sigmas = [x.cpu().detach().numpy() for x in self.sigmas]
        dataset = self.dataset_class(
            LabelGenerator=self.eval_label_generator,
            split=split,
            sample_mode=self.trainer_config.SAMPLER.EVALUATION_SAMPLE_MODE,
            patch_sampler_args=self.eval_dataset_patch_sampling_args,
            dataset_args=self.generic_dataset_args,
            data_aug_args=self.data_aug_args_evaluation,
            label_generator_args=self.label_generator_args,
            sigmas=np_sigmas,
            cache_data=self.trainer_config.TRAINER.CACHE_DATA,
            num_res_supervisions=self.num_res_supervision,
            debug=self.trainer_config.SAMPLER.DEBUG,
            input_size=load_im_size,
            to_pytorch=self.to_pytorch_tensor

        )
        return dataset

    def generate_heatmaps_batch(self, data_dict, dataloader):
        """Generate heatmaps from the main thread. Used only when regressing sigmas, because we can't update the sigma values in the dataloader workers.
            This is a workaround to allow us to update the sigmas in the main thread, and then generate the heatmaps in the main thread.

        Args:
            data_dict (dict): List of dictionaries of samples to generate heatmap labels from.
            dataloader (Dataloader): Dataloader where the generate_labels function is defined.

        Returns:
            batch_hms: The batch of heatmaps generated from the data_dict to use as target labels.
        """
        batch_hms = []
        np_sigmas = [x.cpu().detach().numpy() for x in self.sigmas]
        b_ = [
            dataloader.dataset.generate_labels(x, np_sigmas)
            for x in data_dict["target_coords"]
        ]
        for x in b_:
            if batch_hms == []:
                batch_hms = [[y] for y in x]
            else:
                for hm_idx, hm in enumerate(x):
                    batch_hms[hm_idx].append(hm)
        batch_hms = [torch.stack(x) for x in batch_hms]

        return batch_hms

    @ staticmethod
    def worker_init_fn(worker_id):
        """Function to set the seed for each worker. This is used to ensure that each worker has a different seed,
        so that the random sampling is different for each worker.

        Args:
            worker_id (int): dataloader worker id
        """
        imgaug.seed(np.random.get_state()[1][0] + worker_id)

    def maybe_alter_batch_size(self, dataset, batch_size):
        """If the batch size is set to -1, then we use the entire dataset as the batch size.
            This is used when using GPs since we cannot mini-batch the GP training.
        """

        if batch_size == -1:
            return dataset.__len__()
        else:
            return batch_size

    def save_heatmaps(self, heatmaps):
        hm_dict = {"final_heatmaps": []}
        for idx, results_dict in enumerate(heatmaps['individual_results']):
            if "final_heatmaps" in results_dict.keys():
                hm_dict["final_heatmaps"].append([results_dict["uid"]+"_eval_phase", results_dict["final_heatmaps"]])
        self.dict_logger.log_dict_to_comet(self.comet_logger, hm_dict, -1)
