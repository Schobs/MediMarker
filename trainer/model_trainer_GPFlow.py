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
        self.optimizer = torch.optim.Adam
        self.optimizer_kwargs = {"lr": self.initial_lr}

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
        num_inducing_points = 100
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
        self.logger.info(self.network.parameters)

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

        optimizer = gpf.optimizers.Scipy()
        optimizer.minimize(
            self.network.training_loss_closure((self.all_training_input, self.all_training_labels)),
            variables=self.network.trainable_variables,
            method="l-bfgs-b",
            options={"disp": 50, "maxiter": 10000},
        )
        self.logger.info(self.network)
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

        #     # We validate every 200 epochs
        #     if self.epoch % self.validate_every == 0:
        #         self.logger.info("validation, %s", self.epoch)

        #         with torch.no_grad():
        #             self.network.eval()
        #             generator = iter(self.valid_dataloader)
        #             while generator != None:
        #                 l, generator = self.run_iteration(
        #                     generator,
        #                     self.valid_dataloader,
        #                     backprop=False,
        #                     split="validation",
        #                     log_coords=True,
        #                     logged_vars=per_epoch_logs,
        #                     restart_dataloader=False
        #                 )

        #     self.epoch_end_time = time()

        #     continue_training = self.on_epoch_end(per_epoch_logs)

        #     if not continue_training:
        #         if self.profiler:
        #             self.profiler.stop()
        #         break

        #     self.epoch += 1

    def get_coords_from_heatmap(self, model_output, original_image_size):
        """Gets x,y coordinates from a model output. Here we use the final layer prediction of the U-Net,
            maybe resize and get coords as the peak pixel. Also return value of peak pixel.

        Args:
            output: model output - a stack of heatmaps

        Returns:
            [int, int]: predicted coordinates
        """
        prediction = torch.round(model_output.mean)
        lower, upper = model_output.confidence_region()

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
