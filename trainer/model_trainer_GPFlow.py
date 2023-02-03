from losses.GPLosses import GPFlowLoss
from transforms.generate_labels import GPFlowLabelGenerator
from trainer.model_trainer_base import NetworkTrainer
import copy
from time import time
import gpytorch

from losses.losses import GPLoss
from models.gp_model import ExactGPModel
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

gpf.config.set_default_float(np.float64)
gpf.config.set_default_summary_fmt("notebook")

# torch.multiprocessing.set_start_method('spawn')# good solution !!!!


class GPFlowTrainer(NetworkTrainer):
    """Class for the u-net trainer stuff."""

    def __init__(self, **kwargs):

        super(GPFlowTrainer, self).__init__(**kwargs)

        # global config variable
        # self.early_stop_patience = 250

        # Label generator
        self.train_label_generator = self.eval_label_generator = GPFlowLabelGenerator()

        # get model config parameters

        # scheduler, initialiser and optimiser params
        self.optimizer = tf.keras.optimizers.Adam()

        # "Loss" for GPs - the marginal log likelihood
        self.loss_func = gpytorch.mlls.ExactMarginalLogLikelihood

# likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
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

        self.training_data = None
        self.all_training_input = None
        self.all_training_labels = None

    def initialize_network(self):

        if self.trainer_config.TRAINER.INFERENCE_ONLY:
            self.set_training_dataloaders()

        # self.logger.info(print_summary(self.network))
        # Must initialize model with all training input and labels.
        # We have made sure the batch_size is the dataset.len() for GP so one next() gets the whole dataset.
        self.logger.info("Loading training data for the GP...")
        self.training_data = next(iter(self.train_dataloader))

        self.training_data["image"] = (self.training_data["image"])
        self.training_data["label"]["landmarks"] = np.squeeze(
            self.training_data["label"]["landmarks"])

        self.all_training_input = np.squeeze(np.array(self.training_data["image"], dtype=np.float64), axis=1)
        self.all_training_labels = np.squeeze(np.array(self.training_data["label"]["landmarks"], dtype=np.float64))

        self.logger.info("Initializing GP model with training data...")

        # create multi-output kernel
        kernel = gpf.kernels.SharedIndependent(
            gpf.kernels.SquaredExponential() + gpf.kernels.Linear(), output_dim=2
        )
        # initialization of inducing input locations (M random points from the training inputs)
        flattened_sample_size = self.all_training_input.shape[1]
        num_inducing_points = 500
        Zinit = np.tile(np.linspace(0, flattened_sample_size, num_inducing_points)[:, None], flattened_sample_size)
        Z = Zinit.copy()
        # create multi-output inducing variables from Z
        iv = gpf.inducing_variables.SharedIndependentInducingVariables(
            gpf.inducing_variables.InducingPoints(Z)
        )

        # create SVGP model as usual and optimize
        self.network = gpf.models.SVGP(
            kernel, gpf.likelihoods.Gaussian(), inducing_variable=iv, num_latent_gps=2
        )
        # self.logger.info(self.network.parameters)

        # Log network and initial weights
        if self.comet_logger:
            self.comet_logger.set_model_graph(str(self.network))
            self.logger.info("Logged the model graph.")

    def initialize_optimizer_and_scheduler(self):
        pass

    def initialize_loss_function(self):
        self.loss = GPFlowLoss(self.loss_func)

    # Override the train function from model_trainer_base.py
    def train(self):
        if not self.was_initialized:
            self.initialize(True)

        continue_training = True
        while self.epoch < self.max_num_epochs and continue_training:

            loss = self.network.training_loss_closure((self.all_training_input,
                                                      self.all_training_labels))

            self.logger.info("L: %s", np.array([loss()])[0])
            self.optimizer.minimize(loss,  self.network.trainable_variables)

            self.epoch += 1
            # We validate every 200 epochs
            if self.epoch % self.validate_every == 0:

                val_generator = iter(self.valid_dataloader)
                self.validate(val_generator)
                self.logger.info("validation, %s", self.epoch)

            # continue_training = self.on_epoch_end(per_epoch_logs)

    def validate(self, dataloader):

        for s_idx, data_dict in enumerate(dataloader):

            images = np.squeeze(np.array(data_dict["image"], dtype=np.float64), axis=1)
            labels = np.squeeze(np.array(data_dict["label"]["landmarks"], dtype=np.float64))

            y_mean, y_covar = self.network.predict_y(images, full_cov=True)
            x = 1
          # Log info from this iteration.
            # if list(logged_vars.keys()) != []:
            #     with torch.no_grad():

            #         (
            #             pred_coords,
            #             pred_coords_input_size,
            #             extra_info,
            #             target_coords,
            #         ) = self.maybe_get_coords(output, log_coords, data_dict)

            #         logged_vars = self.dict_logger.log_key_variables(
            #             logged_vars,
            #             pred_coords,
            #             extra_info,
            #             target_coords,
            #             loss_dict,
            #             data_dict,
            #             log_coords,
            #             split,
            #         )
            #         if debug:
            #             debug_vars = [
            #                 x
            #                 for x in logged_vars["individual_results"]
            #                 if x["uid"] in data_dict["uid"]
            #             ]
            #             self.eval_label_generator.debug_prediction(
            #                 data_dict,
            #                 output,
            #                 pred_coords,
            #                 pred_coords_input_size,
            #                 debug_vars,
            #                 extra_info,

        # optimizer.minimize(
        #     self.network.training_loss_closure((self.all_training_input, self.all_training_labels)),
        #     variables=self.network.trainable_variables,
        #     method="l-bfgs-b",
        #     options={"disp": 50, "maxiter": 10000},
        # )
        # self.logger.info(self.network)
        # print_summary(self.network)
        # continue_training = True
        # while self.epoch < self.max_num_epochs and continue_training:

        #     self.epoch_start_time = time()
        #     self.network.train()

        #     per_epoch_logs = self.dict_logger.get_epoch_logger()

        #     # We pass in the entire self.training_data, and we tell it not to restart the dataloader.
        #     # These will do one iteration over the entire dataset.
        #     l, _ = self.run_iteration(
        #         None,
        #         None,
        #         backprop=True,
        #         split="training",
        #         log_coords=False,
        #         logged_vars=per_epoch_logs,
        #         direct_data_dict=self.training_data,
        #         restart_dataloader=False
        #     )

        #     if self.comet_logger:
        #         self.comet_logger.log_metric("training loss", l, self.epoch)
        #         self.comet_logger.log_metric("noise", self.network.likelihood.noise.item(), self.epoch)

        #     if not continue_training:
        #         if self.profiler:
        #             self.profiler.stop()
        #         break

        #     self.epoch += 1

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

        data = (data_dict["image"])

        output = self.network(data)
        del data

        # Only attempts loss if annotations avaliable for entire batch
        if all(data_dict["annotation_available"]):
            loss = self.network.training_loss_closure((self.all_training_input,
                                                      self.all_training_labels))

            loss_dict = {"all_loss_all": loss.numpy(), }
            if backprop:
                self.optimizer.minimize(loss,  self.network.trainable_variables)

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
            pred_coords_input_size, extra_info = self.get_coords_from_heatmap(data_dict)
            pred_coords, target_coords = self.maybe_rescale_coords(
                pred_coords_input_size, data_dict
            )
        else:
            pred_coords = extra_info = target_coords = pred_coords_input_size = None

        return pred_coords, pred_coords_input_size, extra_info, target_coords

    def get_coords_from_heatmap(self, data_dict):
        """Gets x,y coordinates from a model output. Here we use the final layer prediction of the U-Net,
            maybe resize and get coords as the peak pixel. Also return value of peak pixel.

        Args:
            output: model output - a stack of heatmaps

        Returns:
            [int, int]: predicted coordinates
        """

        y_mean, y_var = self.network.predict_y(data_dict["images"], full_cov=True)

        prediction = torch.round(y_mean.mean)
        lower = y_mean - 1.96 * np.sqrt(y_var)
        upper = y_mean + 1.96 * np.sqrt(y_var)

        cov_matr = model_output.covariance_matrix.cpu().detach().numpy()
        extra_info = {"lower": lower, "upper": upper, "cov_matr": cov_matr}
        return prediction,  extra_info

    def stitch_heatmap(self, patch_predictions, stitching_info, gauss_strength=0.5):
        """
        Use model outputs from a patchified image to stitch together a full resolution heatmap

        """

        raise NotImplementedError(
            "need to have original image size passed in because no longer assuming all have same size. see model base trainer for inspo"
        )
