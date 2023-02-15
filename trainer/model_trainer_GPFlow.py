from __future__ import annotations
from ast import Dict, Tuple
import pickle
from typing import Any, Generator, Optional
from losses.GPLosses import GPFlowLoss
from transforms.generate_labels import GPFlowLabelGenerator
from trainer.model_trainer_base import NetworkTrainer
import copy
from time import time
import gpytorch

from losses.losses import GPLoss
import torch
import numpy as np
import tensorflow as tf

# from dataset import ASPIRELandmarks
# import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import os
from scipy.stats import multivariate_normal
import gpflow as gpf
from gpflow.ci_utils import reduce_in_tests
from gpflow.utilities import print_summary
from torch.utils.data import DataLoader
from utils.setup.argument_utils import checkpoint_loading_checking
from models.gp_models.tf_gpmodels import get_SVGP_model, get_conv_SVGP, get_conv_SVGP_linear_coreg


gpf.config.set_default_float(np.float64)
gpf.config.set_default_summary_fmt("notebook")
gpf.config.set_default_jitter(1e-4)
# torch.multiprocessing.set_start_method('spawn')# good solution !!!!


class GPFlowTrainer(NetworkTrainer):
    """Class for the GPFLow trainer."""

    def __init__(self, **kwargs):

        super(GPFlowTrainer, self).__init__(**kwargs)

        # global config variable
        # self.early_stop_patience = 250

        # Label generator
        self.train_label_generator = self.eval_label_generator = GPFlowLabelGenerator()

        # get model config parameters

        # scheduler, initialiser and optimiser params
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.initial_lr)

        self.num_inducing_points = self.trainer_config.MODEL.GPFLOW.NUM_INDUCING_POINTS
        ################# Settings for saving checkpoints ##################################
        # self.save_every = 25

        # override dataloaderbatch size

        self.likelihood = None

        self.data_aug_args_training = {"data_augmentation_strategy": self.trainer_config.SAMPLER.DATA_AUG,
                                       "data_augmentation_package": self.trainer_config.SAMPLER.DATA_AUG_PACKAGE
                                       }
        self.data_aug_args_evaluation = {"data_augmentation_strategy":  self.trainer_config.SAMPLER.DATA_AUG,
                                         "data_augmentation_package": self.trainer_config.SAMPLER.DATA_AUG_PACKAGE
                                         }

        self.kern = self.trainer_config.MODEL.GPFLOW.KERN

        self.training_data = None
        self.all_training_input = None
        self.all_training_labels = None

    def initialize_network(self):
        """
        Initialize the GP model based on the selected kernel.

        vbnet

        If the `kern` attribute is set to "conv", the function will initialize a convolutional GP model.
        If the `kern` attribute is set to "matern52", "se", or "rbf", the function will initialize a standard GP model.

        Returns:
            None
        """

        if self.trainer_config.TRAINER.INFERENCE_ONLY:
            self.set_training_dataloaders()

        self.logger.info("Initializing GP model...")

        if self.kern == "conv":
            all_train_image = [tf.squeeze(tf.convert_to_tensor((x["image"]), dtype=tf.float64), axis=0)
                               for x in self.train_dataloader.dataset]
            all_train_label = [tf.squeeze(tf.convert_to_tensor((x["label"]["landmarks"]),
                                                               dtype=tf.float64), axis=0) for x in self.train_dataloader.dataset]

            self.network = get_conv_SVGP(all_train_image, all_train_label,
                                         self.trainer_config.SAMPLER.PATCH.SAMPLE_PATCH_SIZE, self.num_inducing_points,
                                         self.trainer_config.MODEL.GPFLOW.CONV_KERN_SIZE)

            # self.network = get_conv_SVGP_linear_coreg(all_train_image, all_train_label,
            #                                           self.trainer_config.SAMPLER.PATCH.SAMPLE_PATCH_SIZE, self.num_inducing_points,
            #                                           self.trainer_config.MODEL.GPFLOW.CONV_KERN_SIZE)

            del all_train_image
            del all_train_label
        else:
            if self.kern == "matern52":
                # Mater52 kernel
                kern_list = [
                    gpf.kernels.Matern52() + gpf.kernels.Linear() for _ in range(2)
                ]
            elif self.kern == "se":
                # squared exponential
                kern_list = [
                    gpf.kernels.SquaredExponential() + gpf.kernels.Linear() for _ in range(2)
                ]
            elif self.kern == "rbf":
                # Rbf kernel
                kern_list = [
                    gpf.kernels.SquaredExponential() + gpf.kernels.Linear() for _ in range(2)
                ]
            else:
                raise ValueError("Kernel not implemented. Try 'matern52', 'se' or 'rbf")

            flattened_sample_size = next(iter(self.train_dataloader))["image"].shape[-1]

            self.network = get_SVGP_model(flattened_sample_size, self.num_inducing_points,
                                          kern_list, inducing_dist="uniform")

        # Log network and initial weights
        if self.comet_logger:
            self.comet_logger.set_model_graph(str(self.network))
            self.logger.info("Logged the model graph.")

    def initialize_optimizer_and_scheduler(self):
        pass

    def initialize_loss_function(self):
        self.loss_func = lambda x, y: tf.reduce_mean(tf.square(x - y))

        self.loss = GPLoss(self.loss_func)

    # Override the train function from model_trainer_base.py
    def train(self) -> None:
        """
        Trains the network for a given number of epochs or until continue_training is False.
        The training and validation losses are logged every validate_every epochs.

        :return: None
        """
        if not self.was_initialized:
            self.initialize(True)
        step = 0
        continue_training = True
        while self.epoch < self.max_num_epochs and continue_training:
            self.epoch_start_time = time()
            epoch_loss = 0
            mb_step = 0
            per_epoch_logs = self.dict_logger.get_epoch_logger()

            generator = iter(self.train_dataloader)
            for data_dict in iter(generator):

                l, generator = self.run_iteration(
                    generator,
                    self.train_dataloader,
                    backprop=True,
                    split="training",
                    log_coords=False,
                    logged_vars=per_epoch_logs,
                    direct_data_dict=data_dict,
                    restart_dataloader=False
                )
                epoch_loss += l
                step += 1
                mb_step += 1
                if self.comet_logger:
                    self.comet_logger.log_metric("training loss step",  l, step)

            if self.comet_logger:
                self.comet_logger.log_metric("training loss epoch",  epoch_loss/mb_step, self.epoch)
            # We validate every 200 epochs
            if self.epoch % self.validate_every == 0:

                generator = iter(self.valid_dataloader)
                while generator is not None:
                    l, generator = self.run_iteration(
                        generator,
                        self.valid_dataloader,
                        backprop=False,
                        split="validation",
                        log_coords=True,
                        logged_vars=per_epoch_logs,
                        restart_dataloader=False
                    )

            self.epoch_end_time = time()

            continue_training = self.on_epoch_end(per_epoch_logs)

            self.epoch += 1

    def run_iteration(
        self,
        generator: Generator[Dict[str, Any], None, None],
        dataloader: DataLoader,
        backprop: bool,
        split: str,
        log_coords: bool,
        logged_vars: Optional[Dict[str, Any]] = None,
        debug: bool = False,
        direct_data_dict: Optional[Dict[str, Any]] = None,
        restart_dataloader: bool = True,
    ) -> Tuple[float, Generator[Dict[str, Any], None, None]]:
        """
        Runs a single iteration of a forward pass. It can perform back-propagation (training) or not (validation, testing), indicated w/ bool backprop.
        It can also retrieve coordindates from model output and log variables using DictLogger, dictated by the keys in the logged_vars dict.
        It will generate a batch of samples from the generator, unless a direct_data_dict is provided, in which case it will use that instead.

        Parameters:
        - self: The object instance
        - generator (Generator[Dict[str, Any], None, None]): A generator that produces data dictionaries.
        - dataloader (DataLoader): The dataloader to use if the generator reaches a StopIteration exception.
        - backprop (bool): Whether or not to perform backpropagation.
        - split (str): The split (e.g. train, validation) of the data.
        - log_coords (bool): Whether to extract and log coordinates from the model output.
        - logged_vars (Optional[Dict[str, Any]]):  A Dictionary with keys to log, derived from a template in the class DictLogger. Defaults to None.
        - debug (bool): Whether or not to run in debug mode, defaults to False.
        - direct_data_dict (Optional[Dict[str, Any]]): If not None, will directly perform forward pass on this rather than iterate the generator. Defaults to None.
        - restart_dataloader (bool): Whether or not to restart the dataloader, defaults to True.

        Returns:
        - Tuple[float, Generator[Dict[str, Any], None, None]]: A tuple with the loss from the iteration and the updated generator.
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

        data_dict["image"] = tf.squeeze(tf.convert_to_tensor((data_dict["image"]), dtype=tf.float64), axis=1)
        data_dict["label"]["landmarks"] = tf.squeeze(tf.convert_to_tensor(
            (data_dict["label"]["landmarks"]), dtype=tf.float64), axis=1)

        data = data_dict["image"]
        target = data_dict["label"]["landmarks"]

        # Only attempts loss if annotations avaliable for entire batch
        if all(data_dict["annotation_available"]):
            l = self.optimization_step((data, target), backprop)
            loss_dict = {"all_loss_all": l.numpy()}
        else:
            l = 0
            loss_dict = {}

        # Log info from this iteration.
        if list(logged_vars.keys()) != []:
            with torch.no_grad():

                (
                    pred_coords,
                    pred_coords_input_size,
                    extra_info,
                    target_coords,
                ) = self.maybe_get_coords(log_coords, data_dict)

                if pred_coords is not None:
                    pred_coords = torch.tensor(pred_coords)
                if target_coords is not None:
                    target_coords = torch.tensor(target_coords)
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
                    self.eval_label_generator.debug_prediction_gpflow(
                        data_dict,
                        pred_coords,
                        pred_coords_input_size,
                        debug_vars,
                        extra_info,
                    )

        if self.profiler:
            self.profiler.step()

        return l.numpy(), generator

    def optimization_step(self, batch: Tuple, backprop: bool) -> float:
        """Performs a single optimization step.

        Args:
            batch: A tuple representing the mini-batch of data.
            backprop: A boolean indicating whether to perform backpropagation.

        Returns:
            The value of the loss function.
        """

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.network.trainable_variables)
            loss = self.network.training_loss(batch)
        if backprop:
            grads = tape.gradient(loss, self.network.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.network.trainable_variables))
        return loss

    def maybe_get_coords(self, log_coords: Optional[bool], data_dict: Dict) -> Tuple:
        """Retrieve predicted and target coordinates from data.

        Args:
            log_coords (Optional[bool]): A flag indicating whether to retrieve coordinates.
            data_dict (Dict): A dictionary containing data.

        Returns:
            Tuple: A tuple of predicted coordinates, size of predicted coordinates input,
            additional information, and target coordinates.
        """

        if log_coords:
            pred_coords_input_size, extra_info = self.get_coords_from_heatmap(data_dict)
            pred_coords, target_coords = self.maybe_rescale_coords(
                pred_coords_input_size, data_dict
            )
            target_coords = target_coords.cpu().detach().numpy()
        else:
            pred_coords = extra_info = target_coords = pred_coords_input_size = None

        return pred_coords, pred_coords_input_size, extra_info, target_coords

    def get_coords_from_heatmap(self, data_dict: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Gets x,y coordinates from a model's output.

        Args:
            data_dict (Dict[str, Any]): A dictionary containing image data.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: A tuple of the predicted coordinates and additional information.
        """

        y_mean, cov_matr = self.network.posterior().predict_f(data_dict["image"], full_cov=True, full_output_cov=True)

        prediction = np.expand_dims(np.round(y_mean), axis=1)
        # lower = y_mean - 1.96 * np.array([np.sqrt(cov_matr[x, 0, x, 0]) for x in range(y_mean.shape[0])])
        # upper = y_mean - 1.96 * [np.sqrt(cov_matr[x, 1, x, 1]) for x in range(y_mean.shape[0])]

        extra_info = {"cov_matr": cov_matr}
        return prediction,  extra_info

    def stitch_heatmap(self, patch_predictions, stitching_info, gauss_strength=0.5):
        """
        Use model outputs from a patchified image to stitch together a full resolution heatmap

        """

        raise NotImplementedError(
            "need to have original image size passed in because no longer assuming all have same size. see model base trainer for inspo"
        )

    def on_epoch_end(self, per_epoch_logs: Dict[str, float]) -> bool:
        """
        This function is called at the end of each epoch to log and update some important variables
        related to the training process, check for early stopping conditions, and save a checkpoint if necessary.

        Args:
            per_epoch_logs (Dict[str, float]): A dictionary of metrics logged for the current epoch.

        Returns:
            bool: Indicates whether to continue training or not.
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
            0,
        )

        # log them else they are lost!
        if self.comet_logger:
            self.dict_logger.log_dict_to_comet(
                self.comet_logger, per_epoch_logs, self.epoch
            )

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

        return continue_training

    def save_checkpoint(self, path: str):
        """Save model checkpoint to `path`.

        Args:
            path (str): Path to save model checkpoint to.

        Returns:
            None
        """

        state = {
            "epoch": self.epoch + 1,
            "best_valid_loss": self.best_valid_loss,
            "best_valid_coord_error": self.best_valid_coord_error,
            "best_valid_loss_epoch": self.best_valid_loss_epoch,
            "best_valid_coords_epoch": self.best_valid_coords_epoch,
            "epochs_wo_improvement": self.epochs_wo_val_improv,
            "sigmas": self.sigmas,
            "training_sampler": self.sampler_mode,
            "training_resolution": self.training_resolution,
        }
        with open(path+'meta_state.pkl', 'wb') as f:
            pickle.dump(state, f)

        log_dir = path
        ckpt = tf.train.Checkpoint(model=self.network, optimizer=self.optimizer)
        manager = tf.train.CheckpointManager(ckpt, log_dir, max_to_keep=5)
        manager.save()

    def load_checkpoint(self, model_path: str, training_bool: bool):
        """Load checkpoint from path.

        Args:
            model_path (str): path to checkpoint
            training_bool (bool): If training or not.
        """

        if not self.was_initialized:
            self.initialize(training_bool)

        checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.network)
        manager = tf.train.CheckpointManager(
            checkpoint, model_path, max_to_keep=5)
        status = checkpoint.restore(manager.latest_checkpoint)

        with open(model_path+'meta_state.pkl', 'rb') as f:
            checkpoint_info = pickle.load(f)

        self.epoch = checkpoint_info["epoch"]
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

    # @abstractmethod

    def run_inference(self, split: str, debug: bool = False) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Run inference on the model using the specified data split.

        Parameters:
        -----------
        split: str
            Specify the split of data to be used for inference, e.g. 'training', 'validation', 'test'.
        debug: bool
            Optional parameter to specify whether the run should be in debug mode. Default is False.

        Returns:
        --------
        Tuple[Dict[str, float], Dict[str, float]]
            A tuple containing two dictionaries representing the summary results and individual results of the inference.
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

        # then iterate through dataloader and save to log
        generator = iter(test_dataloader)
        if inference_full_image:
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
            del generator
            print()
        else:
            # this is where we patchify and stitch the input image
            raise NotImplementedError()

        summary_results, ind_results = self.evaluation_metrics(
            evaluation_logs["individual_results"], evaluation_logs["landmark_errors"]
        )

        return summary_results, ind_results