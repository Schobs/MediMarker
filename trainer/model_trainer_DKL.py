from __future__ import annotations
from ast import Dict, Tuple
import pickle
from typing import Any, Generator, Optional

from models.dkl.cigp import CIGP
from transforms.generate_labels import GPFlowLabelGenerator
from trainer.model_trainer_base import NetworkTrainer
from time import time

from losses.losses import GPLoss
import torch
import numpy as np

# from dataset import ASPIRELandmarks
# import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy.stats import multivariate_normal

from torch.utils.data import DataLoader
from utils.im_utils.visualisation import multi_variate_hm
from utils.setup.argument_utils import checkpoint_loading_checking

from trainer.model_trainer_unet import UnetTrainer

from models.UNet_Classic import UNet
# torch.multiprocessing.set_start_method('spawn')# good solution !!!!


class DKLTrainer(UnetTrainer):
    """Class for the GPFLow trainer."""

    def __init__(self, **kwargs):

        super(DKLTrainer, self).__init__(**kwargs)

        # global config variable
        # self.early_stop_patience = 250

        # Label generator
        self.train_label_generator = self.eval_label_generator = GPFlowLabelGenerator()
        super().initialize_network()

        # self.initialize_network()

    def initialize_network(self):
        """Initialise the network."""

        if self.trainer_config.TRAINER.INFERENCE_ONLY:
            self.set_training_dataloaders()

        all_train_image = torch.tensor(np.array([
            (x["image"]) for x in self.train_dataloader.dataset
        ])).to(self.device)
        all_train_label = torch.tensor(np.array([
            (x["label"]["landmarks"]) for x in self.train_dataloader.dataset
        ])).to(self.device)

        assert all_train_label.shape[1] == 1, "Only 1 landmark is supported right now"
        all_train_label = all_train_label.squeeze(1)
        # Let's make the network
        # self, X, Y, dkl_model, normal_y_mode=0)
        self.network = CIGP(all_train_image, all_train_label, self.network, self.device)
        self.network.to(self.device)

        # Log network and initial weights
        if self.comet_logger:
            self.comet_logger.set_model_graph(str(self.network))
            print("Logged the model graph.")

        print(
            "Initialized network architecture. #parameters: ",
            sum(p.numel() for p in self.network.parameters()),
        )

        # Log network and initial weights
        if self.comet_logger:
            self.comet_logger.set_model_graph(str(self.network))
            self.logger.info("Logged the model graph.")

    def initialize_optimizer_and_scheduler(self):
        """Initialise the optimiser and scheduler."""
        assert self.network is not None, "self.initialize_network must be called first"

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.initial_lr)
        # self.optimizer = self.optimizer(self.learnable_params, **self.optimizer_kwargs)

        pass

    def initialize_loss_function(self):

        self.loss = self.network.negative_log_likelihood

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

            # Start by logging validation before any training.
            if self.epoch == 0:
                backprop_train = False
            else:
                backprop_train = True

            generator = iter(self.train_dataloader)
            for data_dict in iter(generator):

                l, generator = self.run_iteration(
                    generator,
                    self.train_dataloader,
                    backprop=backprop_train,
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

            # We validate every X epochs
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

        # data_dict["image"] = tf.squeeze(tf.convert_to_tensor((data_dict["image"]), dtype=tf.float64), axis=1)
        # data_dict["label"]["landmarks"] = tf.squeeze(
        #     tf.convert_to_tensor((data_dict["label"]["landmarks"]), dtype=tf.float64), axis=1
        # )

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
                ) = self.maybe_get_coords(log_coords, log_heatmaps, data_dict)

                if pred_coords is not None and not torch.is_tensor(pred_coords):
                    pred_coords = torch.tensor(pred_coords)
                if target_coords is not None and not torch.is_tensor(target_coords):
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
        self.optimizer.zero_grad()

        loss = self.network.negative_log_likelihood()
        loss.backward()
        self.optimizer.step()

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

            # check here if landmarks unstandardized and target_coords manually unstandardized are the same please.

            extra_info["target_coords_input_size"] = data_dict["label"]["landmarks"]

            pred_coords, target_coords = self.maybe_rescale_coords(pred_coords_input_size, data_dict)
            if not torch.is_tensor(target_coords):
                target_coords = torch.tensor(target_coords)
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

        # y_mean, cov_matr = self.network.posterior().predict_f(data_dict["image"], full_cov=False, full_output_cov=True)
        y_mean, cov_matr = self.network(data_dict["image"])
        # get noise

        prediction = np.expand_dims((y_mean.numpy()), axis=1)

        if self.standardize_landmarks:
            prediction = self.unstandardize_coords(prediction)
            new_cov_list = []
            std_x = self.train_dataloader.dataset.standardize_std[0]
            std_y = self.train_dataloader.dataset.standardize_std[1]
            unstandardized_multiplier = np.array([[std_x**2, std_x*std_y], [std_x*std_y, std_y**2]])
            for x in range(len(y_mean)):
                new_cov_list.append(
                    cov_matr[x] * unstandardized_multiplier)
            cov_matr = torch.stack(new_cov_list)

            # if isinstance(noise, list):
            #     noise = list(noise * np.array([std_x**2, std_y**2]))
            # else:
            #     noise = noise * std_x*std_y

        y_mean_unstandardized = np.squeeze(prediction, axis=1)

        # Need to add the corner point here before scaling. found in data_dict["x_y_corner"]
        if self.inference_eval_mode == "scale_pred_coords":
            x_y_corner = np.expand_dims(
                np.array(
                    [
                        [data_dict["x_y_corner"][0][x], data_dict["x_y_corner"][1][x]]
                        for x in range(len(data_dict["x_y_corner"][0]))
                    ]
                ),
                axis=1,
            )

            prediction = np.add(prediction, x_y_corner)

        extra_info = {}

        extra_info["kernel_cov_matr"] = np.array([cov_matr[x] for x in range(len(y_mean))])
        # extra_info["likelihood_noise"] = np.array([noise] * len(y_mean))

        extra_info["pred_coords_input_size"] = y_mean.numpy()

        if log_heatmaps:
            (
                extra_info["final_heatmaps"],
                extra_info["final_heatmaps_wo_like_noise"],
                extra_info["full_cov_matrix"],
            ) = multi_variate_hm(
                data_dict,
                y_mean_unstandardized,
                cov_matr,
                self.trainer_config.SAMPLER.PATCH.SAMPLE_PATCH_SIZE,
                noise=None,
                plot_targ=True,
                plot_wo_noise_extra=not self.is_train,
            )

        return prediction, extra_info

    def stitch_heatmap(self, patch_predictions, stitching_info, gauss_strength=0.5):
        """
        Use model outputs from a patchified image to stitch together a full resolution heatmap

        """

        raise NotImplementedError(
            "need to have original image size passed in because no longer assuming all have same size. see model base trainer for inspo"
        )

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

            target_coords = np.round(data_dict["target_coords"]["unstandardized"])  #
            # C3 (and C1 if input size == full res size so full & target the same)

        # C2
        if self.use_full_res_coords and not self.resize_first:

            # If we are using patch_centred sampling, we are basing prediction on a smaller patch
            # containing the landmark, so we need to add the patch corner to the prediction.
            if not torch.is_tensor(pred_coords):
                pred_coords = torch.tensor(pred_coords).to(self.device)

            upscale_factor = data_dict["resizing_factor"].to(self.device)

            upscaled_coords = torch.mul(pred_coords, upscale_factor)
            pred_coords = torch.round(upscaled_coords).to(self.device)
            # pred_coords = pred_coords * upscale_factor

        return pred_coords, target_coords

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
        elif self.lr_policy == "scheduled_3000" and self.epoch == 3000:
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
        for idx, results_dict in enumerate(heatmaps["individual_results"]):
            if "final_heatmaps" in results_dict.keys():
                hm_dict["final_heatmaps"].append([results_dict["uid"] + "_eval_phase", results_dict["final_heatmaps"]])
            if "final_heatmaps_wo_like_noise" in results_dict.keys():
                hm_dict["final_heatmaps_wo_like_noise"].append(
                    [results_dict["uid"] + "_eval_phase_nolike", results_dict["final_heatmaps_wo_like_noise"]]
                )
        self.dict_logger.log_dict_to_comet(self.comet_logger, hm_dict, -1)

    def unstandardize_coords(self, stand_coords):
        lm_mean = self.train_dataloader.dataset.standardize_mean
        lm_std = self.train_dataloader.dataset.standardize_std

        return np.multiply(stand_coords, lm_std) + (lm_mean)
