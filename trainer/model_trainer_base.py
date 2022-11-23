import os
import types
from inference.ensemble_inference_helper import EnsembleUncertainties
import torch
import numpy as np
from time import time

# import multiprocessing as mp
from utils.local_logging.dict_logger import DictLogger
from torch.cuda.amp import GradScaler, autocast
from evaluation.localization_evaluation import (
    success_detection_rate,
    generate_summary_df,
)
from torch.utils.data import DataLoader

from abc import ABC, abstractmethod
import imgaug
import pandas as pd


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
        self.is_train = is_train

        # Dataset class to use
        self.dataset_class = dataset_class
        # Dataloader info
        self.data_loader_batch_size = self.trainer_config.SOLVER.DATA_LOADER_BATCH_SIZE
        self.num_batches_per_epoch = self.trainer_config.SOLVER.MINI_BATCH_SIZE
        self.gen_hms_in_mainthread = (
            self.trainer_config.INFERRED_ARGS.GEN_HM_IN_MAINTHREAD
        )
        self.sampler_mode = self.trainer_config.SAMPLER.SAMPLE_MODE
        self.landmarks = self.trainer_config.DATASET.LANDMARKS
        self.training_resolution = (
            self.trainer_config.SAMPLER.PATCH.RESOLUTION_TO_SAMPLE_FROM
            if self.sampler_mode == "patch"
            else self.trainer_config.SAMPLER.INPUT_SIZE
        )

        # Set up logger & profiler
        self.profiler = profiler
        self.comet_logger = comet_logger
        self.verbose_logging = self.trainer_config.OUTPUT.VERBOSE

        # Set up directories
        self.output_folder = output_folder

        # Trainer variables
        self.perform_validation = self.trainer_config.TRAINER.PERFORM_VALIDATION
        self.fold = self.trainer_config.TRAINER.FOLD
        self.continue_checkpoint = self.trainer_config.MODEL.CHECKPOINT

        self.auto_mixed_precision = self.trainer_config.SOLVER.AUTO_MIXED_PRECISION

        # Training params
        self.max_num_epochs = self.trainer_config.SOLVER.MAX_EPOCHS
        # Regressing sigma parameters for heatmaps
        self.regress_sigma = self.trainer_config.SOLVER.REGRESS_SIGMA
        self.sigmas = [
            torch.tensor(x, dtype=float, device=self.device, requires_grad=True)
            for x in np.repeat(
                self.trainer_config.MODEL.GAUSS_SIGMA, len(self.landmarks)
            )
        ]

        # Validation parameters
        self.use_full_res_coords = self.trainer_config.INFERRED_ARGS.USE_FULL_RES_COORDS
        self.resize_first = self.trainer_config.INFERRED_ARGS.RESIZE_FIRST

        # Checkpointing params
        self.save_every = 25
        self.save_latest_only = (
            self.trainer_config.TRAINER.SAVE_LATEST_ONLY
        )  # if false it will not store/overwrite _latest but separate files each
        self.save_intermediate_checkpoints = (
            True  # whether or not to save checkpoint_latest
        )

        # Loss function

        # To be initialised in the super class (here)
        self.was_initialized = False
        self.amp_grad_scaler = None
        self.train_dataloader = self.valid_dataloader = None
        self.dict_logger = None
        self.learnable_params = None

        # Set up in init of extended class (child)
        self.network = types.SimpleNamespace()  # empty object
        self.train_label_generator = self.eval_label_generator = None
        self.optimizer = None
        self.loss = types.SimpleNamespace()  # empty object
        self.num_res_supervision = 1

        # Can be changed in extended class (child)
        self.early_stop_patience = 150
        self.initial_lr = self.trainer_config.SOLVER.BASE_LR

        # Inference params
        self.fit_gauss_inference = self.trainer_config.INFERENCE.FIT_GAUSS

        # Initialize
        self.epoch = 0
        self.best_valid_loss = 999999999999999999999999999
        self.best_valid_coord_error = 999999999999999999999999999
        self.best_valid_loss_epoch = 0
        self.epochs_wo_val_improv = 0
        self.print_initiaization_info = True
        self.epoch_start_time = time()
        self.epoch_end_time = time()

    def initialize(self, training_bool=True):
        """
        Initialize profiler, comet logger, training/val dataloaders, network, optimizer, loss, automixed precision
        and maybe load a checkpoint.

        """
        # torch.backends.cudnn.benchmark = True

        if self.profiler:
            print("Initialized profiler")
            self.profiler.start()

        self.initialize_dataloader_settings()
        if training_bool:
            self.set_training_dataloaders()

        self.initialize_network()
        self.initialize_optimizer_and_scheduler()
        self.initialize_loss_function()
        self._maybe_init_amp()

        if self.comet_logger:
            self.comet_logger.log_parameters(self.trainer_config)

        # This is the logger that will save epoch results to log & log variables at inference, extend this for any extra stuff you want to log/save at evaluation!
        self.dict_logger = DictLogger(
            len(self.landmarks),
            self.regress_sigma,
            self.loss.loss_seperated_keys,
            self.dataset_class.additional_sample_attribute_keys,
        )

        self.was_initialized = True

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
            print(msg)

    @abstractmethod
    def get_coords_from_heatmap(self, model_output, original_image_size):

        """
        Function to take model output and return coordinates & a Dict of any extra information to log (e.g. max of heatmap)
        """

    def train(self):
        """
        The main training loop. For every epoch we train and validate. Each training epoch covers a number of minibatches of a certain batch size,
        defined in the config file.
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

            print("training")
            # Train for X number of batches per epoch e.g. 250
            for _ in range(self.num_batches_per_epoch):
                l, generator = self.run_iteration(
                    generator,
                    self.train_dataloader,
                    backprop=True,
                    split="training",
                    log_coords=False,
                    logged_vars=per_epoch_logs,
                )
                if self.comet_logger:
                    self.comet_logger.log_metric("training loss iteration", l, step)
                step += 1
            # del generator
            print("validation")

            with torch.no_grad():
                self.network.eval()
                generator = iter(self.valid_dataloader)
                while generator != None:
                    l, generator = self.run_iteration(
                        generator,
                        self.valid_dataloader,
                        backprop=False,
                        split="validation",
                        log_coords=True,
                        logged_vars=per_epoch_logs,
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
            print("Logging weights as histogram...")
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

        so = time()

        # We can either give the generator to be iterated or a data_dict directly
        if direct_data_dict is None:
            try:
                data_dict = next(generator)

            except StopIteration:
                if split != "training":
                    return 0, None
                else:
                    generator = iter(dataloader)
                    data_dict = next(generator)
        else:
            data_dict = direct_data_dict

        data = (data_dict["image"]).to(self.device)

        # This happens when we regress sigma with >0 workers due to multithreading issues.
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
        so = time()
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
        s = time()
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

        e = time()
        if self.profiler:
            self.profiler.step()

        del output
        del target
        return l.detach().cpu().numpy(), generator

    def maybe_get_coords(self, output, log_coords, data_dict):
        """From output gets coordinates and extra info for logging. If log_coords is false, returns None for all.
            It also decides whether to resize heatmap, rescale coords depending on config settings.

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
            pred_coords, target_coords = self.maybe_rescale_coords(
                pred_coords_input_size, data_dict
            )
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

        print("Epoch %s logs: %s" % (self.epoch, per_epoch_logs))

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
            print(
                "EARLY STOPPING. Validation Coord Error did not reduce for %s epochs. "
                % self.early_stop_patience
            )

        self.maybe_save_checkpoint(new_best_valid, new_best_coord_valid)

        self.maybe_update_lr(epoch=self.epoch)

        return continue_training

    def maybe_save_checkpoint(self, new_best_valid_bool, new_best_valid_coord_bool):
        """
        Saves a checkpoint every save_ever epochs.
        :return:
        """

        fold_str = str(self.fold)
        if (
            self.save_intermediate_checkpoints
            and (self.epoch % self.save_every == (self.save_every - 1))
        ) or self.epoch == self.max_num_epochs - 1:
            print("saving scheduled checkpoint file...")
            if not self.save_latest_only:
                self.save_checkpoint(
                    os.path.join(
                        self.output_folder,
                        "model_ep_" + str(self.epoch) + "_fold" + fold_str + ".model",
                    )
                )

                if self.epoch >= 150:
                    self.save_every = 50
                if self.epoch >= 250:
                    self.save_every = 100
            self.save_checkpoint(
                os.path.join(
                    self.output_folder, "model_latest_fold" + (fold_str) + ".model"
                )
            )
            print("done")
        if new_best_valid_bool:
            print(
                "saving scheduled checkpoint file as it's new best on validation set..."
            )
            self.save_checkpoint(
                os.path.join(
                    self.output_folder,
                    "model_best_valid_loss_fold" + fold_str + ".model",
                )
            )

            print("done")

        if new_best_valid_coord_bool:
            print(
                "saving scheduled checkpoint file as it's new best on validation set for coord error..."
            )
            self.save_checkpoint(
                os.path.join(
                    self.output_folder,
                    "model_best_valid_coord_error_fold" + fold_str + ".model",
                )
            )

            print("done")

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
        # C1 or C2
        if self.use_full_res_coords:
            target_coords = data_dict["full_res_coords"].to(self.device)  # C1
        else:
            target_coords = np.round(data_dict["target_coords"]).to(
                self.device
            )  # C3 (and C1 if input size == full res size so full & target the same)

        # C2
        if self.use_full_res_coords and not self.resize_first:
            upscale_factor = torch.tensor(
                [data_dict["resizing_factor"][0], data_dict["resizing_factor"][1]]
            ).to(self.device)
            # upscaled_coords = torch.tensor([pred_coords[x]*upscale_factor[x] for x in range(len(pred_coords))]).to(self.device)
            upscaled_coords = torch.mul(pred_coords, upscale_factor)
            pred_coords = torch.round(upscaled_coords)
            # pred_coords = pred_coords * upscale_factor

        return pred_coords, target_coords

    # @abstractmethod
    def patchify_and_predict(self, single_sample, logged_vars):
        """Function that takens in a large input image, patchifies it and runs each patch through the model & stitches heatmap together

        #1) should split up into patches of given patch-size.
        #2) should run patches through in batches using run_iteration, NOT LOGGING ANYTHING but needs to return the OUTPUTS somehow.
            MUST ADD OPTION TO RETURN OUTPUTS in run_iteration?
        #3) Need to use method to stitch patches together (future, phdnet will use patch size 512 512 for now).
        #4) call log_key_variables function now with the final big heatmap as the "output". The logging should work as usual from that.

        Returns:
            _type_: _description_
        """

    # @abstractmethod
    def run_inference(self, split, debug=False):
        """Function to run inference on a full sized input

        #0) instantitate test dataset and dataloader
        #1A) if FULL:
            i) iterate dataloader and run_iteration each time to go through and save results.
            ii) should run using run_iteration with logged_vars to log
        1B) if PATCHIFYING full_res_output  <- this can be fututre addition
            i) use patchify_and_predict to stitch hm together with logged_vars to log

        #2) need a way to deal with the key dictionary & combine all samples
        #3) need to put evaluation methods in evluation function & import and ues key_dict for analysis
        #4) return individual results & do summary results.
        """

        # If trained using patch, return the full image, else ("full") will return the image size network was trained on.
        if self.sampler_mode == "patch":
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
        self.test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.data_loader_batch_size,
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
        generator = iter(self.test_dataloader)
        if inference_full_image:
            while generator != None:
                print("-", end="")
                l, generator = self.run_iteration(
                    generator,
                    self.test_dataloader,
                    backprop=False,
                    split=split,
                    log_coords=True,
                    logged_vars=evaluation_logs,
                    debug=debug,
                )
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
        if self.sampler_mode == "patch":
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
        self.test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.data_loader_batch_size,
            shuffle=False,
            num_workers=0,
            worker_init_fn=NetworkTrainer.worker_init_fn,
        )

        # Initialise ensemble postprocessing and uncertainty estimation
        uncertainty_estimation_keys = (
            self.trainer_config.INFERENCE.ENSEMBLE_UNCERTAINTY_KEYS
        )
        smha_model_idx = self.trainer_config.INFERENCE.UNCERTAINTY_SMHA_MODEL_IDX

        self.ensemble_handler = EnsembleUncertainties(
            uncertainty_estimation_keys, smha_model_idx, self.landmarks
        )

        # network evaluation mode
        self.network.eval()

        # Initialise esenmble results dictionaries
        ensemble_result_dicts = {
            uncert_key: [] for uncert_key in self.ensemble_handler.uncertainty_keys
        }
        all_ind_errors = {
            uncert_key: [[] for x in range(len(self.landmarks))]
            for uncert_key in self.ensemble_handler.uncertainty_keys
        }

        # Iterate through dataloader and save to log
        generator = iter(self.test_dataloader)
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
                            self.test_dataloader,
                            backprop=False,
                            split=split,
                            log_coords=True,
                            logged_vars=evaluation_logs,
                            debug=debug,
                            direct_data_dict=direct_data_dict,
                        )

                    # Analyse batch for s-mha, e-mha, and e-cpv and maybe errors (if we have annotations)
                    (
                        ensembles_analyzed,
                        ind_landmark_errors,
                    ) = self.ensemble_handler.ensemble_inference_with_uncertainties(
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

        self.checkpoint_loading_checking()

        if self.auto_mixed_precision:
            self._maybe_init_amp()

            if "amp_grad_scaler" in checkpoint_info.keys():
                self.amp_grad_scaler.load_state_dict(checkpoint_info["amp_grad_scaler"])

        if self.print_initiaization_info:
            print("Loaded checkpoint %s. Epoch: %s, " % (model_path, self.epoch))

    def checkpoint_loading_checking(self):
        """Checks that the loaded checkpoint is compatible with the current model and training settings.

        Raises:
            ValueError: _description_
            ValueError: _description_
        """

        # Check the sampler in config is the same as the one in the checkpoint.
        if self.sampler_mode != self.trainer_config.SAMPLER.SAMPLE_MODE:
            raise ValueError(
                "model was trained using SAMPLER.SAMPLE_MODE %s but attempting to load with SAMPLER.SAMPLE_MODE %s. \
                Please amend this in config file."
                % (self.sampler_mode, self.trainer_config.SAMPLER.SAMPLE_MODE)
            )

        # check if the training resolution from config is the same as the one in the checkpoint.
        if self.sampler_mode == "patch":
            if (
                self.training_resolution
                != self.trainer_config.SAMPLER.PATCH.RESOLUTION_TO_SAMPLE_FROM
            ):
                raise ValueError(
                    "model was trained using SAMPLER.PATCH.RESOLUTION_TO_SAMPLE_FROM %s but attempting to load with self.training_resolution %s. \
                    Please amend this in config file."
                    % (self.training_resolution, self.training_resolution)
                )
        else:
            if self.training_resolution != self.trainer_config.SAMPLER.INPUT_SIZE:
                raise ValueError(
                    "model was trained using SAMPLER.INPUT_SIZE %s but attempting to load with self.training_resolution %s. \
                    Please amend this in config file."
                    % (self.training_resolution, self.trainer_config.SAMPLER.INPUT_SIZE)
                )

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

        train_dataset = self.dataset_class(
            annotation_path=self.trainer_config.DATASET.SRC_TARGETS,
            landmarks=self.landmarks,
            LabelGenerator=self.train_label_generator,
            split="training",
            sample_mode=self.sampler_mode,
            sample_patch_size=self.trainer_config.SAMPLER.PATCH.SAMPLE_PATCH_SIZE,
            sample_patch_bias=self.trainer_config.SAMPLER.PATCH.SAMPLER_BIAS,
            sample_patch_from_resolution=self.trainer_config.SAMPLER.PATCH.RESOLUTION_TO_SAMPLE_FROM,
            root_path=self.trainer_config.DATASET.ROOT,
            sigmas=np_sigmas,
            generate_hms_here=not self.gen_hms_in_mainthread,
            cv=self.fold,
            cache_data=self.trainer_config.TRAINER.CACHE_DATA,
            num_res_supervisions=self.num_res_supervision,
            debug=self.trainer_config.SAMPLER.DEBUG,
            input_size=self.trainer_config.SAMPLER.INPUT_SIZE,
            hm_lambda_scale=self.trainer_config.MODEL.HM_LAMBDA_SCALE,
            data_augmentation_strategy=self.trainer_config.SAMPLER.DATA_AUG,
            data_augmentation_package=self.trainer_config.SAMPLER.DATA_AUG_PACKAGE,
            dataset_split_size=self.trainer_config.DATASET.TRAINSET_SIZE,
        )

        if self.perform_validation:
            # if patchify, we want to return the full image
            if self.sampler_mode == "patch":
                valid_dataset = self.get_evaluation_dataset(
                    "validation",
                    self.trainer_config.SAMPLER.PATCH.RESOLUTION_TO_SAMPLE_FROM,
                )
            else:
                valid_dataset = self.get_evaluation_dataset(
                    "validation", self.trainer_config.SAMPLER.INPUT_SIZE
                )

        else:
            if self.sampler_mode == "patch":
                valid_dataset = self.get_evaluation_dataset(
                    "training",
                    self.trainer_config.SAMPLER.PATCH.RESOLUTION_TO_SAMPLE_FROM,
                    dataset_split_size=self.trainer_config.DATASET.TRAINSET_SIZE,
                )
            else:
                valid_dataset = self.get_evaluation_dataset(
                    "training",
                    self.trainer_config.SAMPLER.INPUT_SIZE,
                    dataset_split_size=self.trainer_config.DATASET.TRAINSET_SIZE,
                )
            print(
                'WARNING: NOT performing validation. Instead performing "validation" on training set for coord error metrics.'
            )

        print(
            "Using %s Dataloader workers and persist workers bool : %s "
            % (self.num_workers_cfg, self.persist_workers)
        )
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.data_loader_batch_size,
            shuffle=True,
            num_workers=self.num_workers_cfg,
            persistent_workers=self.persist_workers,
            worker_init_fn=NetworkTrainer.worker_init_fn,
            pin_memory=True,
        )
        self.valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=self.data_loader_batch_size,
            shuffle=False,
            num_workers=self.num_workers_cfg,
            persistent_workers=self.persist_workers,
            worker_init_fn=NetworkTrainer.worker_init_fn,
            pin_memory=True,
        )

    def get_evaluation_dataset(self, split, load_im_size, dataset_split_size=-1):
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
            annotation_path=self.trainer_config.DATASET.SRC_TARGETS,
            landmarks=self.landmarks,
            LabelGenerator=self.eval_label_generator,
            split=split,
            sample_mode=self.trainer_config.SAMPLER.EVALUATION_SAMPLE_MODE,
            sample_patch_size=self.trainer_config.SAMPLER.PATCH.SAMPLE_PATCH_SIZE,
            sample_patch_bias=self.trainer_config.SAMPLER.PATCH.SAMPLER_BIAS,
            sample_patch_from_resolution=self.trainer_config.SAMPLER.PATCH.RESOLUTION_TO_SAMPLE_FROM,
            root_path=self.trainer_config.DATASET.ROOT,
            sigmas=np_sigmas,
            generate_hms_here=not self.gen_hms_in_mainthread,
            cv=self.fold,
            cache_data=self.trainer_config.TRAINER.CACHE_DATA,
            num_res_supervisions=self.num_res_supervision,
            debug=self.trainer_config.SAMPLER.DEBUG,
            data_augmentation_strategy=None,
            input_size=load_im_size,
            hm_lambda_scale=self.trainer_config.MODEL.HM_LAMBDA_SCALE,
            dataset_split_size=dataset_split_size,
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

    @staticmethod
    def worker_init_fn(worker_id):
        """Function to set the seed for each worker. This is used to ensure that each worker has a different seed,
        so that the random sampling is different for each worker.

        Args:
            worker_id (int): dataloader worker id
        """
        imgaug.seed(np.random.get_state()[1][0] + worker_id)
