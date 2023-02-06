from ast import Tuple
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
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

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
        # self.logger.info("Loading training data for the GP...")
        # self.training_data = self.train_dataloader.dataset
        # self.valid_dataset = self.valid_dataloader.dataset

        # # get batch sizes
        # train_batch_size = self.maybe_alter_batch_size(self.train_dataloader, self.data_loader_batch_size_train)
        # valid_batch_size = self.maybe_alter_batch_size(self.valid_dataloader, self.data_loader_batch_size_eval)

        # self.training_data["image"] = np.squeeze(np.array(self.training_data["image"], dtype=np.float64), axis=1)
        # self.training_data["label"]["landmarks"] = np.squeeze(
        #     np.array(self.training_data["label"]["landmarks"], dtype=np.float64))
        # self.tf_training_data = tf.data.Dataset.from_tensor_slices(
        #     (self.training_data["image"],  self.training_data["label"]["landmarks"]))
        # self.tf_training_data = self.tf_training_data.batch()

        # self.all_valid_input = np.array([(x["image"][0]) for x in self.valid_dataloader.dataset], dtype=np.float64)
        # self.all_valid_labels = np.array([(x["label"]["landmarks"][0])
        #                                  for x in self.valid_dataloader.dataset], dtype=np.float64)
        # self.tf_validation_data = tf.data.Dataset.from_tensor_slices((self.all_valid_input, self.tf_validation_data))

        # self.all_training_input = np.squeeze(np.array(self.training_data["image"], dtype=np.float64), axis=1)

        # self.all_training_input = self.training_data["image"]
        # self.all_training_labels = self.training_data["label"]["landmarks"]

        self.logger.info("Initializing GP model with training data...")

        # create multi-output kernel
        kern_list = [
            gpf.kernels.SquaredExponential() + gpf.kernels.Linear() for _ in range(2)
        ]
        # Create multi-output kernel from kernel list
        kernel = gpf.kernels.LinearCoregionalization(
            kern_list, W=np.random.randn(2, 2)
        )  # Notice that we initialise the mixing matrix W

        # kernel = gpf.kernels.SharedIndependent(
        #     gpf.kernels.SquaredExponential() + gpf.kernels.Linear(), output_dim=2
        # )
        # initialization of inducing input locations (M random points from the training inputs)
        flattened_sample_size = next(iter(self.train_dataloader))[0].shape[1]
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
            self.epoch_start_time = time()

            per_epoch_logs = self.dict_logger.get_epoch_logger()

            # We pass in the entire self.training_data, and we tell it not to restart the dataloader.
            # These will do one iteration over the entire dataset.
            generator = iter(self.train_dataloader)
            l, _ = self.run_iteration(
                generator,
                self.train_dataloader,
                backprop=True,
                split="training",
                log_coords=False,
                logged_vars=per_epoch_logs,
                # direct_data_dict=self.training_data,
                restart_dataloader=False
            )

            self.logger.info("Training Loss, %s", l)

            if self.comet_logger:
                self.comet_logger.log_metric("training loss",  l, self.epoch)

            # We validate every 200 epochs
            if self.epoch % self.validate_every == 0:
                self.logger.info("validation, %s", self.epoch)

                generator = iter(self.valid_dataloader)
                while generator != None:
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
            # loss = self.network.training_loss_closure((self.all_training_input,
            #                                           self.all_training_labels))

            # self.logger.info("L: %s", np.array([loss()])[0])
            # self.optimizer.minimize(loss,  self.network.trainable_variables)

            # self.epoch += 1
            # # We validate every 200 epochs
            # if self.epoch % self.validate_every == 0:

            #     val_generator = iter(self.valid_dataloader)
            #     self.validate(val_generator)
            #     self.logger.info("validation, %s", self.epoch)

            # continue_training = self.on_epoch_end(per_epoch_logs)

    # def validate(self, dataloader):

    #     for s_idx, data_dict in enumerate(dataloader):

    #         images = np.squeeze(np.array(data_dict["image"], dtype=np.float64), axis=1)
    #         labels = np.squeeze(np.array(data_dict["label"]["landmarks"], dtype=np.float64))

    #         y_mean, y_covar = self.network.predict_f(images, full_cov=True, full_output_cov=True)

            # self.logger.info("y_covar: %s", y_covar.numpy())
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

        data_dict["image"] = tf.convert_to_tensor((data_dict["image"]), dtype=tf.float64)
        data_dict["label"]["landmarks"] = tf.convert_to_tensor((data_dict["label"]["landmarks"]), dtype=tf.float64)

        data = data_dict["image"]
        target = data_dict["label"]["landmarks"]

        # Only attempts loss if annotations avaliable for entire batch
        if all(data_dict["annotation_available"]):
            l = self.network.training_loss_closure((data, target))

            loss_dict = {"all_loss_all": np.array([l()])[0]}
            if backprop:
                self.optimizer.minimize(l,  self.network.trainable_variables)

        else:
            l = tf.tensor(0)
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

        return np.array([l()])[0], generator

    def optimization_step(
        self, batch
    ):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.network.trainable_variables)
            loss = self.network.training_loss(batch)
        grads = tape.gradient(loss, self.network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.network.trainable_variables))
        return loss

    def maybe_get_coords(self, log_coords, data_dict):
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
            target_coords = target_coords.cpu().detach().numpy()
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

        y_mean, cov_matr = self.network.predict_f(data_dict["image"][:, 0, :], full_cov=True, full_output_cov=True)

        prediction = np.expand_dims(np.round(y_mean), axis=0)
        lower = y_mean - 1.96 * np.sqrt(cov_matr[0, 0, 0, 0])
        upper = y_mean + 1.96 * np.sqrt(cov_matr[0, 1, 0, 1])

        extra_info = {"lower": lower, "upper": upper, "cov_matr": cov_matr}
        return prediction,  extra_info

    def stitch_heatmap(self, patch_predictions, stitching_info, gauss_strength=0.5):
        """
        Use model outputs from a patchified image to stitch together a full resolution heatmap

        """

        raise NotImplementedError(
            "need to have original image size passed in because no longer assuming all have same size. see model base trainer for inspo"
        )

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

    def save_checkpoint(self, path):
        """Save model checkpoint to path

        Args:
            path (str): Path to save model checkpoint to.
        """
        # state = {
        #     "epoch": self.epoch + 1,
        #     "state_dict": self.network.state_dict(),
        #     "optimizer": self.optimizer.state_dict(),
        #     "best_valid_loss": self.best_valid_loss,
        #     "best_valid_coord_error": self.best_valid_coord_error,
        #     "best_valid_loss_epoch": self.best_valid_loss_epoch,
        #     "best_valid_coords_epoch": self.best_valid_coords_epoch,
        #     "epochs_wo_improvement": self.epochs_wo_val_improv,
        #     "sigmas": self.sigmas,
        #     "training_sampler": self.sampler_mode,
        #     "training_resolution": self.training_resolution,
        # }

        log_dir = path
        ckpt = tf.train.Checkpoint(model=self.network)
        manager = tf.train.CheckpointManager(ckpt, log_dir, max_to_keep=3)
        manager.save()

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
            patch_sampler_args=self.dataset_patch_sampling_args,
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
            img_resolution = self.trainer_config.SAMPLER.PATCH.RESOLUTION_TO_SAMPLE_FROM,
        else:
            img_resolution = self.trainer_config.SAMPLER.INPUT_SIZE

        valid_dataset = self.get_evaluation_dataset(validation_split, img_resolution)

        #### Create DataLoaders ####

        self.logger.info("Using %s Dataloader workers and persist workers bool : %s ",
                         self.num_workers_cfg, self.persist_workers)

        train_batch_size = self.maybe_alter_batch_size(train_dataset, self.data_loader_batch_size_train)
        valid_batch_size = self.maybe_alter_batch_size(valid_dataset, self.data_loader_batch_size_eval)

        all_train_input = np.array([(x["image"][0]) for x in train_dataset], dtype=np.float64)
        all_train_labels = np.array([(x["label"]["landmarks"][0]) for x in train_dataset], dtype=np.float64)

        self.train_dataloader = tf.data.Dataset.from_tensor_slices((all_train_input,  all_train_labels))
        self.train_dataloader = self.train_dataloader.batch(train_batch_size)

        all_valid_input = np.array([(x["image"][0]) for x in valid_dataset], dtype=np.float64)
        all_valid_labels = np.array([(x["label"]["landmarks"][0]) for x in valid_dataset], dtype=np.float64)
        self.valid_dataloader = tf.data.Dataset.from_tensor_slices((all_valid_input, all_valid_labels))
        self.valid_dataloader = self.valid_dataloader.batch(valid_batch_size)
