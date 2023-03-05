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
from utils.im_utils.visualisation import multi_variate_hm
from utils.setup.argument_utils import checkpoint_loading_checking
from models.gp_models.register import *
from models.gp_models.tf_gpmodels import (
    conv_sgp_rbf_fix,
    get_SVGP_model,
    get_conv_SVGP,
    get_conv_SVGP_linear_coreg,
    toms,
)

gpf.config.set_default_float(np.float64)
gpf.config.set_default_summary_fmt("notebook")
gpf.config.set_default_jitter(1e-3)
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

        self.data_aug_args_training = {
            "data_augmentation_strategy": self.trainer_config.SAMPLER.DATA_AUG,
            "data_augmentation_package": self.trainer_config.SAMPLER.DATA_AUG_PACKAGE,
            "guarantee_lms_image": self.trainer_config.SAMPLER.DATA_AUG_GUARANTEE_LMS_IN_IMAGE,
        }
        self.data_aug_args_evaluation = {
            "data_augmentation_strategy": self.trainer_config.SAMPLER.DATA_AUG,
            "data_augmentation_package": self.trainer_config.SAMPLER.DATA_AUG_PACKAGE,
            "guarantee_lms_image": self.trainer_config.SAMPLER.DATA_AUG_GUARANTEE_LMS_IN_IMAGE,
        }

        self.kern = self.trainer_config.MODEL.GPFLOW.KERN
        self.kern_stride = self.trainer_config.MODEL.GPFLOW.CONV_KERN_STRIDE
        self.ip_sample_var = self.trainer_config.MODEL.GPFLOW.INDUCING_SAMPLE_VAR
        self.conv_kern_ls = self.trainer_config.MODEL.GPFLOW.CONV_KERN_LS  # base kernel lengthscale
        self.conv_kern_var = self.trainer_config.MODEL.GPFLOW.CONV_KERN_V  # base kernel variance
        self.conv_kern_type = self.trainer_config.MODEL.GPFLOW.CONV_KERN_TYPE
        self.fix_noise_until = self.trainer_config.MODEL.GPFLOW.FIX_NOISE_UNTIL_EPOCH
        self.train_inducing_points = self.trainer_config.MODEL.GPFLOW.TRAIN_IP
        self.initial_likelihood_noise = self.trainer_config.MODEL.GPFLOW.MODEL_NOISE_INIT
        self.independent_likelihoods = self.trainer_config.MODEL.GPFLOW.INDEPENDENT_LIKELIHOODS
        self.likelihood_noise_upper_bound = self.trainer_config.MODEL.GPFLOW.LIKELIHOOD_NOISE_UPPER_BOUND
        self.likelihood_training_intervals = self.trainer_config.MODEL.GPFLOW.LIKELIHOOD_NOISE_TRAINING_INTERVALS
        self.likelihood_seperate_optim = self.trainer_config.MODEL.GPFLOW.LIKELIHOOD_NOISE_SEPERATE_OPTIM
        self.kl_scale = self.trainer_config.MODEL.GPFLOW.KL_SCALE

        self.training_data = None
        self.all_training_input = None
        self.all_training_labels = None

        if self.likelihood_seperate_optim:
            self.logger.info("Using seperate optimiser for likelihood noise")
            self.likelihood_optimizer = tf.keras.optimizers.Adam(learning_rate=self.initial_lr / 100)

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
            all_train_image = [
                tf.squeeze(tf.convert_to_tensor((x["image"]), dtype=tf.float64), axis=0)
                for x in self.train_dataloader.dataset
            ]
            all_train_label = [
                tf.squeeze(tf.convert_to_tensor((x["label"]["landmarks"]), dtype=tf.float64), axis=0)
                for x in self.train_dataloader.dataset
            ]

            # self.network = get_conv_SVGP(all_train_image, all_train_label,
            #                              self.trainer_config.SAMPLER.PATCH.SAMPLE_PATCH_SIZE, self.num_inducing_points,
            #                              self.trainer_config.MODEL.GPFLOW.CONV_KERN_SIZE)

            # self.network = conv_sgp_rbf_fix(all_train_image, all_train_label,
            #                                 self.trainer_config.SAMPLER.PATCH.SAMPLE_PATCH_SIZE, self.num_inducing_points,
            #                                 self.trainer_config.MODEL.GPFLOW.CONV_KERN_SIZE, kern_stride=self.kern_stride)

            self.network = get_conv_SVGP_linear_coreg(
                all_train_image,
                all_train_label,
                self.trainer_config.SAMPLER.PATCH.SAMPLE_PATCH_SIZE,
                self.num_inducing_points,
                self.trainer_config.MODEL.GPFLOW.CONV_KERN_SIZE,
                inducing_sample_var=self.ip_sample_var,
                base_kern_ls=self.conv_kern_ls,
                base_kern_var=self.conv_kern_var,
                init_likelihood_noise=self.initial_likelihood_noise,
                independent_likelihoods=self.independent_likelihoods,
                likelihood_upper_bound=self.likelihood_noise_upper_bound,
                kern_type=self.conv_kern_type,
                kl_scale=self.kl_scale,
            )

            if not self.train_inducing_points:
                self.logger.info("Fixing inducing points. They will not train.")
                gpf.set_trainable(self.network.inducing_variable, False)
            else:
                self.logger.info("Not fixing inducing points, they will train.")

            if self.fix_noise_until > self.epoch:
                self.logger.info("Fixing likelihood variance until Epoch %s" % self.fix_noise_until)
                self._toggle_likelihood_noise_trainable()
                # if self.independent_likelihoods:
                #     [gpf.set_trainable(x.variance, False) for x in self.network.likelihood.likelihoods]
                # else:
                #     gpf.set_trainable(self.network.likelihood.variance, False)
            else:
                self.logger.info(
                    "Not Fixing likelihood variance. Training. Learning independent likelihoods is %s."
                    % self.independent_likelihoods
                )
                self._set_likelihood_noise_trainable()

                # if self.independent_likelihoods:
                #     [gpf.set_trainable(x.variance, True) for x in self.network.likelihood.likelihoods]
                # else:
                #     gpf.set_trainable(self.network.likelihood.variance, True)

            # TODO set this false. add config for #epochs to turn back on try 10,50,100

            # gpf.set_trainable(self.network.likelihood.variance, False)
            # gpf.set_trainable(self.network.inducing_variable, False)

            del all_train_image
            del all_train_label
        else:
            if self.kern == "matern52":
                # Mater52 kernel
                kern_list = [gpf.kernels.Matern52() + gpf.kernels.Linear() for _ in range(2)]
            elif self.kern == "se":
                # squared exponential
                kern_list = [gpf.kernels.SquaredExponential() + gpf.kernels.Linear() for _ in range(2)]
            elif self.kern == "rbf":
                # Rbf kernel
                kern_list = [gpf.kernels.SquaredExponential() + gpf.kernels.Linear() for _ in range(2)]
            else:
                raise ValueError("Kernel not implemented. Try 'matern52', 'se' or 'rbf")

            flattened_sample_size = next(iter(self.train_dataloader))["image"].shape[-1]

            self.network = get_SVGP_model(
                flattened_sample_size, self.num_inducing_points, kern_list, inducing_dist="uniform"
            )

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
                    log_heatmaps=False,
                    logged_vars=per_epoch_logs,
                    direct_data_dict=data_dict,
                    restart_dataloader=False,
                )
                epoch_loss += l
                step += 1
                mb_step += 1
                if self.comet_logger:
                    self.comet_logger.log_metric("training loss step", l, step)

            if self.comet_logger:
                self.comet_logger.log_metric("training loss epoch", epoch_loss / mb_step, self.epoch)
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
                        log_heatmaps=self.validation_log_heatmaps,
                        logged_vars=per_epoch_logs,
                        restart_dataloader=False,
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
        log_heatmaps: bool,
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
        data_dict["label"]["landmarks"] = tf.squeeze(
            tf.convert_to_tensor((data_dict["label"]["landmarks"]), dtype=tf.float64), axis=1
        )

        data = data_dict["image"]
        target = data_dict["label"]["landmarks"]

        # Only attempts loss if annotations avaliable for entire batch
        if all(data_dict["annotation_available"]):
            l = self.optimization_step((data, target), backprop)
            loss_dict = {"all_loss_all": l.numpy()}
            if self.independent_likelihoods:
                loss_dict["x_noise"] = self.network.likelihood.likelihoods[0].variance.numpy()
                loss_dict["y_noise"] = self.network.likelihood.likelihoods[1].variance.numpy()
            else:
                loss_dict["noise"] = self.network.likelihood.variance.numpy()
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
                ) = self.maybe_get_coords(log_coords, log_heatmaps, data_dict)

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
                    debug_vars = [x for x in logged_vars["individual_results"] if x["uid"] in data_dict["uid"]]
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
        if self.likelihood_seperate_optim:
            if self.independent_likelihoods:
                assert (
                    len(self.network.trainable_variables) == 9
                ), "to index the noise params, the network must have 9 trainable variables"
                train_vars_normal_optim = self.network.trainable_variables[:7]
                train_vars_noise_optim = self.network.trainable_variables[7:]
            else:
                assert (
                    len(self.network.trainable_variables) == 8
                ), "to index the noise param, the network must have 8 trainable variables"
                train_vars_normal_optim = self.network.trainable_variables[:7]
                train_vars_noise_optim = self.network.trainable_variables[7:]
        else:
            train_vars_normal_optim = self.network.trainable_variables
            train_vars_noise_optim = []

        # else:

        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(self.network.trainable_variables)
            loss = self.network.training_loss(batch)
        if backprop:
            grads = tape.gradient(loss, train_vars_normal_optim)
            self.optimizer.apply_gradients(zip(grads, train_vars_normal_optim))

            if self.likelihood_seperate_optim:
                grads = tape.gradient(loss, train_vars_noise_optim)
                self.likelihood_optimizer.apply_gradients(zip(grads, train_vars_noise_optim))

        return loss

    def maybe_get_coords(self, log_coords: bool, log_heatmaps: bool, data_dict: Dict) -> Tuple:
        """Retrieve predicted and target coordinates from data.

        Args:
            log_coords (Optional[bool]): A flag indicating whether to retrieve coordinates.
            data_dict (Dict): A dictionary containing data.

        Returns:
            Tuple: A tuple of predicted coordinates, size of predicted coordinates input,
            additional information, and target coordinates.
        """

        if log_coords:
            pred_coords_input_size, extra_info = self.get_coords_from_heatmap(data_dict, log_heatmaps)
            pred_coords, target_coords = self.maybe_rescale_coords(pred_coords_input_size, data_dict)
            if torch.is_tensor(target_coords):
                target_coords = target_coords.cpu().detach().numpy()
        else:
            pred_coords = extra_info = target_coords = pred_coords_input_size = None

        return pred_coords, pred_coords_input_size, extra_info, target_coords

    def get_coords_from_heatmap(
        self, data_dict: Dict[str, Any], log_heatmaps: bool
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Gets x,y coordinates from a model's output.

        Args:
            data_dict (Dict[str, Any]): A dictionary containing image data.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: A tuple of the predicted coordinates and additional information.
        """

        y_mean, cov_matr = self.network.posterior().predict_f(data_dict["image"], full_cov=True, full_output_cov=True)

        # get noise

        if self.independent_likelihoods:
            noise = [x.variance.numpy() for x in self.network.likelihood.likelihoods]
        else:
            noise = self.network.likelihood.variance.numpy()

        prediction = np.expand_dims((y_mean), axis=1)

        # Need to add the corner point here before scaling. found in data_dict["x_y_corner"]
        if self.inference_eval_mode == "scale_pred_coords" and not self.is_train:
            # x_y_corner = np.expand_dims(np.stack(data_dict["x_y_corner"]), axis=1)
            # x_y_corner =
            x_y_corner = np.expand_dims(np.array(
                [[data_dict["x_y_corner"][0][x], data_dict["x_y_corner"][1][x]] for x in range(len(data_dict["x_y_corner"][0]))]), axis=1)

            # x_y_corner = np.expand_dims(np.stack([(np.flip(x.numpy())) for x in data_dict["x_y_corner"]]), axis=1)
            prediction = np.add(prediction, x_y_corner)

        # lower = y_mean - 1.96 * np.array([np.sqrt(cov_matr[x, 0, x, 0]) for x in range(y_mean.shape[0])])
        # upper = y_mean - 1.96 * [np.sqrt(cov_matr[x, 1, x, 1]) for x in range(y_mean.shape[0])]

        extra_info = {"cov_matr": cov_matr}
        if log_heatmaps:
            extra_info["final_heatmaps"], extra_info["final_heatmaps_wo_like_noise"] = multi_variate_hm(
                data_dict,
                y_mean,
                cov_matr,
                self.trainer_config.SAMPLER.PATCH.SAMPLE_PATCH_SIZE,
                noise=noise,
                plot_targ=True,
                plot_wo_noise_extra=not self.is_train
            )

        return prediction, extra_info

    # def maybe_rescale_coords(self, pred_coords, data_dict):
    #     """Maybe rescale coordinates based on evaluation parameters, and decide which target coords to evaluate against.
    #         Cases C1:4:
    #         C1) used full-scale image to train or resized heatmap already: leave predicted, use full-res target
    #         C2) used low-scale image to train and rescaling coordinates up to full-scale image size
    #         C3) use low-scale image to train, want to eval on low-scale coordinates
    #     Args:
    #         pred_coords (tensor): coords extracted from output heatmap
    #         data_dict (dict): dataloader sample dictionary

    #     Returns:
    #         tensor, tensor: predicted coordinates and target coordinates for evaluation
    #     """

    #     # Don't worry in case annotations are not present since these are 0,0 anyway. this is handled elesewhere
    #     # C1 or C2
    #     if self.use_full_res_coords:
    #         target_coords = data_dict["full_res_coords"].to(self.device)  # C1
    #     else:
    #         # C3 (and C1 if input size == full res size so full & target the same)
    #         target_coords = np.round(data_dict["label"]["landmarks"])

    #     # C3
    #     if self.use_full_res_coords and not self.resize_first:
    #         upscale_factor = torch.tensor(
    #             [data_dict["resizing_factor"][0], data_dict["resizing_factor"][1]]
    #         ).to(self.device)
    #         # upscaled_coords = torch.tensor([pred_coords[x]*upscale_factor[x] for x in range(len(pred_coords))]).to(self.device)
    #         upscaled_coords = torch.mul(pred_coords, upscale_factor)
    #         pred_coords = torch.round(upscaled_coords)
    #         # pred_coords = pred_coords * upscale_factor

    #     return pred_coords, target_coords

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
            self.dict_logger.log_dict_to_comet(self.comet_logger, per_epoch_logs, self.epoch)

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
            self.epochs_wo_val_improv = 0
        else:
            self.epochs_wo_val_improv += 1

        if per_epoch_logs["valid_coord_error_mean"] < self.best_valid_coord_error:
            self.best_valid_coord_error = per_epoch_logs["valid_coord_error_mean"]
            self.best_valid_coords_epoch = self.epoch
            new_best_coord_valid = True

        self.check_early_stop()

        self.maybe_save_checkpoint(new_best_valid, new_best_coord_valid)
        self.maybe_update_lr()

        self.fix_unfix_noise()

        # self.maybe_clip_noise()

        return continue_training

    def check_early_stop(self):
        """Checks if early stopping should be triggered and updates model accordingly."""
        if self.epochs_wo_val_improv == self.early_stop_patience:
            is_variance_trainable = True
            if self.independent_likelihoods:
                is_variance_trainable = all([x.variance.trainable for x in self.network.likelihood.likelihoods])
            else:
                is_variance_trainable = self.network.likelihood.variance.trainable

            if not is_variance_trainable and self.likelihood_noise_upper_bound is not None:
                self._set_likelihood_noise_trainable()
                self.logger.info(
                    "No improvement in %s epochs with fixed noise. Unfixing noise and restarting early stopping.",
                    self.early_stop_patience,
                )

                self.epochs_wo_val_improv = 0
            else:
                self.logger.info(
                    "Early stopping triggered. Validation Coord Error did not reduce for %s epochs.",
                    self.early_stop_patience,
                )
                self.continue_training = False

    def fix_unfix_noise(self):
        """
        Make the noise parameter(s) trainable.

        If the likelihood noise parameter is currently fixed, this method makes it trainable
        so that it can be optimized during training.

        If the model has independent likelihoods, it will make all of their noise parameters trainable.
        Otherwise, it will make the noise parameter of the overall likelihood trainable.

        """
        self._maybe_unfix_noise()
        self._maybe_train_likelihood_noise()

    def _maybe_unfix_noise(self):
        """Unfix the noise parameter(s) if the fix_noise_until epoch has been reached."""
        if self.epoch == self.fix_noise_until and self.epoch > 0:
            self.logger.info("Unfixing noise. Now training.")
            self._set_likelihood_noise_trainable()

    def _maybe_train_likelihood_noise(self):
        """Train the likelihood noise parameter(s) if the epoch is in a training interval."""
        if self.epoch > self.fix_noise_until and self.likelihood_training_intervals is not None:
            if self.epoch % self.likelihood_training_intervals == 0:
                self.logger.info("Toggling likelihood noise parameter training.")
                self._toggle_likelihood_noise_trainable()

    def _set_likelihood_noise_trainable(self):
        """Set the noise parameter(s) of the likelihood to trainable."""
        if self.independent_likelihoods:
            for likelihood in self.network.likelihood.likelihoods:
                gpf.set_trainable(likelihood.variance, True)

        else:
            gpf.set_trainable(self.network.likelihood.variance, True)
        self.logger.info("Likelihood noise parameter(s) set to trainable.")

    def _toggle_likelihood_noise_trainable(self):
        """Toggle the training status of the noise parameter(s) of the likelihood."""
        if self.independent_likelihoods:
            is_variance_trainable = all(
                [likelihood.variance.trainable for likelihood in self.network.likelihood.likelihoods]
            )
            for likelihood in self.network.likelihood.likelihoods:
                gpf.set_trainable(likelihood.variance, not is_variance_trainable)
        else:
            is_variance_trainable = self.network.likelihood.variance.trainable
            gpf.set_trainable(self.network.likelihood.variance, not is_variance_trainable)

        self.logger.info(
            "Toggled likelihood noise parameter training. New trainable status: %s", not is_variance_trainable
        )

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

        log_dir = path
        if self.likelihood_seperate_optim:
            ckpt = tf.train.Checkpoint(
                model=self.network, optimizer=self.optimizer, likelihood_optimizer=self.likelihood_optimizer
            )
        else:
            ckpt = tf.train.Checkpoint(model=self.network, optimizer=self.optimizer)
        manager = tf.train.CheckpointManager(ckpt, log_dir, max_to_keep=5)
        manager.save()

        with open(path + "/meta_state.pkl", "wb") as f:
            pickle.dump(state, f)

    def load_checkpoint(self, model_path: str, training_bool: bool):
        """Load checkpoint from path.

        Args:
            model_path (str): path to checkpoint
            training_bool (bool): If training or not.
        """

        if not self.was_initialized:
            self.initialize(training_bool)

        if self.likelihood_seperate_optim:
            checkpoint = tf.train.Checkpoint(
                model=self.network, optimizer=self.optimizer, likelihood_optimizer=self.likelihood_optimizer
            )
        else:
            checkpoint = tf.train.Checkpoint(model=self.network, optimizer=self.optimizer)

        manager = tf.train.CheckpointManager(checkpoint, model_path, max_to_keep=5)
        status = checkpoint.restore(manager.latest_checkpoint)

        with open(model_path + "/meta_state.pkl", "rb") as f:
            checkpoint_info = pickle.load(f)

        self.epoch = checkpoint_info["epoch"]
        if training_bool:
            self.best_valid_loss = checkpoint_info["best_valid_loss"]
            self.best_valid_loss_epoch = checkpoint_info["best_valid_loss_epoch"]
            self.best_valid_coord_error = checkpoint_info["best_valid_coord_error"]
            self.best_valid_coords_epoch = checkpoint_info["best_valid_coords_epoch"]
            self.epochs_wo_val_improv = checkpoint_info["epochs_wo_improvement"]

        # For some reason it loads in as the log of the variance, so we need to convert it back
        if self.independent_likelihoods:
            for likelihood in self.network.likelihood.likelihoods:
                if likelihood.variance.numpy() < 0:
                    likelihood.variance = gpf.Parameter(tf.exp(likelihood.variance.numpy()))
        else:
            if self.network.likelihood.variance.numpy() < 0:
                self.network.likelihood.variance = gpf.Parameter(tf.exp(self.network.likelihood.variance.numpy()))

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
            if self.trainer_config.SAMPLER.PATCH.INFERENCE_MODE == "patchify_and_stitch":
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
                    log_heatmaps=self.inference_log_heatmaps,
                    logged_vars=evaluation_logs,
                    debug=debug,
                    restart_dataloader=False,
                )
            del generator
            print()
        else:
            # this is where we patchify and stitch the input image
            raise NotImplementedError()

        summary_results, ind_results = self.evaluation_metrics(
            evaluation_logs["individual_results"], evaluation_logs["landmark_errors"]
        )

        if self.inference_log_heatmaps:
            self.save_heatmaps(evaluation_logs)

        return summary_results, ind_results

    def maybe_update_lr(self):
        """
        Update the learning rate if the learning rate policy is set to "scheduled_10".

        """
        if self.lr_policy == "scheduled_10" and self.epoch == 10:
            self.logger.info("Reducing LR from %s to %s", self.initial_lr, self.initial_lr / 10)

            self.optimizer.lr.assign(self.initial_lr / 10)

        elif self.lr_policy == "scheduled_250" and self.epoch == 250:
            self.logger.info("Reducing LR from %s to %s", self.initial_lr, self.initial_lr / 10)

        elif self.lr_policy == "scheduled_500" and self.epoch == 500:
            self.logger.info("Reducing LR from %s to %s", self.initial_lr, self.initial_lr / 10)

            self.optimizer.lr.assign(self.initial_lr / 10)
        elif self.lr_policy == "scheduled_1000" and self.epoch == 1000:
            self.logger.info("Reducing LR from %s to %s", self.initial_lr, self.initial_lr / 10)

            self.optimizer.lr.assign(self.initial_lr / 10)

        elif self.lr_policy == "scheduled_2000" and self.epoch == 2000:
            self.logger.info("Reducing LR from %s to %s", self.initial_lr, self.initial_lr / 10)

            self.optimizer.lr.assign(self.initial_lr / 10)
        elif self.lr_policy == "scheduled_100_2000":
            if self.epoch == 100:
                self.logger.info("Reducing LR from %s to %s", self.initial_lr, self.initial_lr / 10)

                self.optimizer.lr.assign(self.initial_lr / 10)
            elif self.epoch == 2000:
                self.logger.info("Reducing LR from %s to %s", self.initial_lr / 10, (self.initial_lr / 100))
                self.optimizer.lr.assign(self.initial_lr / 100)

        elif self.lr_policy == "scheduled_500_3000":
            if self.epoch == 100:
                self.logger.info("Reducing LR from %s to %s", self.initial_lr, self.initial_lr / 10)

                self.optimizer.lr.assign(self.initial_lr / 10)
            elif self.epoch == 2000:
                self.logger.info("Reducing LR from %s to %s", self.initial_lr / 10, (self.initial_lr / 100))
                self.optimizer.lr.assign(self.initial_lr / 100)

    def save_heatmaps(self, heatmaps):
        hm_dict = {"final_heatmaps": [], "final_heatmaps_wo_like_noise": []}
        for idx, results_dict in enumerate(heatmaps['individual_results']):
            if "final_heatmaps" in results_dict.keys():
                hm_dict["final_heatmaps"].append([results_dict["uid"]+"_eval_phase", results_dict["final_heatmaps"]])
            if "final_heatmaps_wo_like_noise" in results_dict.keys():
                hm_dict["final_heatmaps_wo_like_noise"].append(
                    [results_dict["uid"]+"_eval_phase_nolike", results_dict["final_heatmaps_wo_like_noise"]])
        self.dict_logger.log_dict_to_comet(self.comet_logger, hm_dict, -1)
