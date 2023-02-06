from ast import Tuple
import pickle
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
from torch.utils.data import DataLoader
from utils.setup.argument_utils import checkpoint_loading_checking

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

        self.logger.info("Initializing GP model...")

        # create multi-output kernel
        kern_list = [
            gpf.kernels.SquaredExponential() + gpf.kernels.Linear() for _ in range(2)
        ]
        # Create multi-output kernel from kernel list
        kernel = gpf.kernels.LinearCoregionalization(
            kern_list, W=np.random.randn(2, 2)
        )  # Notice that we initialise the mixing matrix W

        flattened_sample_size = next(iter(self.train_dataloader))["image"].shape[-1]
        num_inducing_points = 1000
        Zinit = np.tile(np.linspace(0, flattened_sample_size, num_inducing_points)[:, None], flattened_sample_size)
        Z = Zinit.copy()
        # create multi-output inducing variables from Z
        iv = gpf.inducing_variables.SharedIndependentInducingVariables(
            gpf.inducing_variables.InducingPoints(Z)
        )

        # initialize mean of variational posterior to be of shape MxL
        q_mu = np.zeros((num_inducing_points, 2))
        # initialize \sqrt(Σ) of variational posterior to be of shape LxMxM
        q_sqrt = np.repeat(np.eye(num_inducing_points)[None, ...], 2, axis=0) * 1.0
        # create SVGP model as usual and optimize
        self.network = gpf.models.SVGP(
            kernel, gpf.likelihoods.Gaussian(), inducing_variable=iv, num_latent_gps=2, q_mu=q_mu, q_sqrt=q_sqrt,
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
        step = 0
        continue_training = True
        while self.epoch < self.max_num_epochs and continue_training:
            self.epoch_start_time = time()
            epoch_loss = 0
            mb_step = 0
            per_epoch_logs = self.dict_logger.get_epoch_logger()

            # We pass in the entire self.training_data, and we tell it not to restart the dataloader.
            # These will do one iteration over the entire dataset.
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
                # self.logger.info("Training Step %s, Loss, %s", step, l)
                step += 1
                mb_step += 1
                if self.comet_logger:
                    self.comet_logger.log_metric("training loss step",  l, step)

            if self.comet_logger:
                self.comet_logger.log_metric("training loss epoch",  epoch_loss/mb_step, self.epoch)
            # We validate every 200 epochs
            if self.epoch % self.validate_every == 0:
                # self.logger.info("validation, %s", self.epoch)

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

    def optimization_step(
        self, batch, backprop
    ):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.network.trainable_variables)
            loss = self.network.training_loss(batch)
        if backprop:
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

        y_mean, cov_matr = self.network.predict_f(data_dict["image"], full_cov=True, full_output_cov=True)

        prediction = np.expand_dims(np.round(y_mean), axis=0)
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

    def load_checkpoint(self, model_path, training_bool):
        """Load checkpoint from path.

        Args:
            model_path (str): path to checjpoint
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
