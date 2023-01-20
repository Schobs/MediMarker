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

# torch.multiprocessing.set_start_method('spawn')# good solution !!!!
from trainer.model_trainer_base import NetworkTrainer

from transforms.generate_labels import GPLabelGenerator


class GPTrainer(NetworkTrainer):
    """Class for the u-net trainer stuff."""

    def __init__(self, **kwargs):

        super(GPTrainer, self).__init__(**kwargs)

        # global config variable
        self.early_stop_patience = 250

        # Label generator
        self.train_label_generator = self.eval_label_generator = GPLabelGenerator()

        # get model config parameters

        # scheduler, initialiser and optimiser params
        self.optimizer = torch.optim.Adam
        self.optimizer_kwargs = {"lr": self.initial_lr}

        # "Loss" for GPs - the marginal log likelihood
        self.loss_func = gpytorch.mlls.ExactMarginalLogLikelihood

        ################# Settings for saving checkpoints ##################################
        self.save_every = 25

        # override dataloaderbatch size

        self.likelihood = None

        self.data_aug_args_training = {"data_augmentation_strategy": self.trainer_config.SAMPLER.DATA_AUG,
                                       "data_augmentation_package": self.trainer_config.SAMPLER.DATA_AUG_PACKAGE
                                       }
        self.data_aug_args_evaluation = {"data_augmentation_strategy":  self.trainer_config.SAMPLER.DATA_AUG,
                                         "data_augmentation_package": self.trainer_config.SAMPLER.DATA_AUG_PACKAGE
                                         }

    def initialize_network(self):

        if self.trainer_config.TRAINER.INFERENCE_ONLY:
            self.set_training_dataloaders()

        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=2, rank=2
        )

        # Must initialize model with all training input and labels
        self.logger.info("Loading training data...")
        self.training_data = next(iter(self.train_dataloader))

        self.training_data["image"] = torch.squeeze(self.training_data["image"]).type(torch.float32)
        self.training_data["label"]["landmarks"] = torch.squeeze(
            self.training_data["label"]["landmarks"]).type(torch.float32)

        self.all_training_input = self.training_data["image"].to(self.device)
        self.all_training_labels = self.training_data["label"]["landmarks"].to(self.device)
        # self.all_training_input = torch.squeeze(self.training_data["image"]).type(torch.float32).to(self.device)
        # self.all_training_labels = torch.squeeze(
        #     self.training_data["label"]["landmarks"]).type(torch.float32).to(self.device)

        # self.logger.info("Loading validation data...")
        # self.validation_data = next(iter(self.valid_dataloader))
        # self.all_validation_input = torch.squeeze(self.training_data["image"]).type(torch.float32).to(self.device)
        # self.all_validation_labels = torch.squeeze(
        #     self.validation_data["label"]["landmarks"]).type(torch.float32).to(self.device)

        self.logger.info("Initializing model with training data...")
        self.network = ExactGPModel(self.all_training_input, self.all_training_labels, self.likelihood)
        self.network.to(self.device)

        # Log network and initial weights
        if self.comet_logger:
            self.comet_logger.set_model_graph(str(self.network))
            print("Logged the model graph.")

        print(
            "Initialized network architecture. #parameters: ",
            sum(p.numel() for p in self.network.parameters()),
        )

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"

        self.learnable_params = list(self.network.parameters())
        self.optimizer = self.optimizer(self.learnable_params, **self.optimizer_kwargs)

        print("Initialised optimizer.")

    def initialize_loss_function(self):
        self.loss_func = self.loss_func(self.likelihood, self.network)
        self.loss = GPLoss(self.loss_func)

        print("initialized Loss function.")

    # Override the train function
    def train(self):
        if not self.was_initialized:
            self.initialize(True)

        continue_training = True
        step = 0
        while self.epoch < self.max_num_epochs and continue_training:

            self.epoch_start_time = time()
            self.network.train()

            per_epoch_logs = self.dict_logger.get_epoch_logger()

            l, _ = self.run_iteration(
                None,
                None,
                backprop=True,
                split="training",
                log_coords=False,
                logged_vars=per_epoch_logs,
                direct_data_dict=self.training_data,
                restart_dataloader=False
            )

            if self.comet_logger:
                self.comet_logger.log_metric("training loss iteration", l, step)
            step += 1

            if self.epoch % 200 == 0:
                self.logger.info("validation, %s", self.epoch)

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
                            restart_dataloader=False
                        )

                self.epoch_end_time = time()

                continue_training = self.on_epoch_end(per_epoch_logs)

                if not continue_training:
                    if self.profiler:
                        self.profiler.stop()
                    break

            self.epoch += 1

            # # Zero gradients from previous iteration
            # self.optimizer.zero_grad()
            # # Output from model
            # # self.logger.info("training input shape : %s", self.all_training_input.shape)
            # output = self.network(self.all_training_input)
            # self.logger.info(output)
            # # Calc loss and backprop gradients
            # loss, loss_dict = self.loss(output, self.all_training_labels)

            # loss.backward()
            # self.optimizer.step()
            # self.logger.info(
            #     "Iter %d/%d - Loss: %.3f  noise: %.3f",
            #     self.epoch + 1,
            #     self.max_num_epochs,
            #     loss.item(),
            #     # self.network.covar_module.base_kernel.lengthscale.item(),
            #     self.network.likelihood.noise.item(),
            # )
            # self.epoch += 1

            # self.comet_logger.log_metric("training loss", loss.item(), self.epoch)
            # self.comet_logger.log_metric(
            #     "noise", self.network.likelihood.noise.item(), self.epoch
            # )
            # per_epoch_logs = self.log_iter(output, per_epoch_logs, loss_dict, self.all_training_input, "training")
            # # self.comet_logger.log_metric("training loss", loss, self.epoch)

            # if self.epoch % 50 == 0:
            #     per_epoch_logs = self.validation(self.dict_logger.get_epoch_logger())
            #     continue_training = self.on_epoch_end(per_epoch_logs)

        # self.save_checkpoint(
        #     os.path.join(
        #         self.output_folder, "GP_model_latest__fold" + str(self.generic_dataset_args["fold"]) + ".model"
        #     )
        # )

    # def validation(self, per_epoch_logs):
    #     self.network.eval()
    #     self.likelihood.eval()

    #     # print("inference_input shape: ", self.validation_data.shape)
    #     for sample_idx, data_dict in iter(self.validation_data):
    #         with torch.no_grad(), gpytorch.settings.fast_pred_var():

    #             data = torch.squeeze(data_dict["image"]).type(torch.float32).to(self.device)
    #             target = torch.squeeze(data_dict["label"]["landmarks"]).type(torch.float32).to(self.device)

    #             predictions = self.likelihood(
    #                 self.network(torch.unsqueeze(data, dim=0))
    #             )

    #             # observed_pred = observed_preds[sample_idx]

    #             mean = predictions.mean.cpu().detach().numpy()
    #             lower, upper = predictions.confidence_region()

    #             cov_matr = predictions.covariance_matrix.cpu().detach().numpy()
    #             err = np.linalg.norm(mean - target)

    #             self.logger.info(
    #                 "predictions for sample %s: %s and label %s with error %s",
    #                 sample_idx, mean, target, err
    #             )

    #             self.logger.info(
    #                 "mean shape %s and cov shape %s : ",
    #                 mean.shape,
    #                 cov_matr.shape,
    #             )
    #             self.logger.info("mean %s and cov %s", mean, cov_matr)

    #             per_epoch_logs = self.log_iter(mean, per_epoch_logs, {}, data_dict,
    #                                            "validation", log_coords=True, pred_coords=mean)

    #     # all_val_errors.append(err)
    #     return per_epoch_logs

    # def log_iter(self, output, logged_vars, loss_dict, data_dict, split, debug=False,  log_coords=False, pred_coords=None):
    #     # Log info from this iteration.
    #     if list(logged_vars.keys()) != []:
    #         with torch.no_grad():

    #             (
    #                 pred_coords,
    #                 pred_coords_input_size,
    #                 extra_info,
    #                 target_coords,
    #             ) = self.maybe_get_coords(output, log_coords, data_dict)

    #             logged_vars = self.dict_logger.log_key_variables(
    #                 logged_vars,
    #                 pred_coords,
    #                 extra_info,
    #                 target_coords,
    #                 loss_dict,
    #                 data_dict,
    #                 log_coords,
    #                 split,
    #             )
    #             if debug:
    #                 debug_vars = [
    #                     x
    #                     for x in logged_vars["individual_results"]
    #                     if x["uid"] in data_dict["uid"]
    #                 ]
    #                 self.eval_label_generator.debug_prediction(
    #                     data_dict,
    #                     output,
    #                     pred_coords,
    #                     pred_coords_input_size,
    #                     debug_vars,
    #                     extra_info,
    #                 )
    #     return logged_vars

    # def run_inference(self, split, debug=False):
    #     raise NotImplementedError()
    #     self.network.eval()
    #     self.likelihood.eval()

    #     # print("observed_preds: ", observed_preds.shape)
    #     # Test points are regularly spaced along [0,1]
    #     # Make predictions by feeding model through likelihood
    #     # inference_input = self.all_training_input[0].reshape(1,-1)

    #     self.logger.info("Loading Testing data...")
    #     self.testing = next(iter(self.test_dataloader))
    #     self.all_validation_input = torch.squeeze(self.training_data["image"]).type(torch.float32).to(self.device)
    #     self.all_validation_labels = torch.squeeze(
    #         self.validation_data["label"]["landmarks"]).type(torch.float32).to(self.device)
    #     inference_input = self.all_testing_input
    #     inference_labels = self.all_testing_labels
    #     inference_ims = self.all_testing_input_ims
    #     print("inference_input shape: ", inference_input.shape)
    #     print("training input shape: ", self.all_training_input.shape)
    #     # print("all training ims shape: ", self.all_training_input_ims.shape)
    #     # with torch.no_grad(), gpytorch.settings.fast_pred_var():
    #     #     predictions = self.likelihood(self.network(inference_input))

    #     #     # observed_pred = observed_preds[sample_idx]

    #     #     mean = predictions.mean.cpu().detach().numpy()
    #     #     lower, upper = predictions.confidence_region()

    #     #     cov_matr = predictions.covariance_matrix.cpu().detach().numpy()

    #     #     # first_sample_cv = [[cov_matr[0,0], cov_matr[0,1]], [cov_matr[1,0], cov_matr[1,1]]]

    #     #     print("mean shape: ", mean.shape)

    #     #     print("cov matrx shape: ", cov_matr.shape)
    #     #     print("cov matrx: ", cov_matr)

    #     # print("all attributes: ", dir(predictions))

    #     for sample_idx in range(len(inference_input)):
    #         with torch.no_grad(), gpytorch.settings.fast_pred_var():
    #             predictions = self.likelihood(
    #                 self.network(torch.unsqueeze(inference_input[sample_idx], dim=0))
    #             )

    #             # observed_pred = observed_preds[sample_idx]

    #             mean = predictions.mean.cpu().detach().numpy()
    #             lower, upper = predictions.confidence_region()

    #             cov_matr = predictions.covariance_matrix.cpu().detach().numpy()
    #             print(
    #                 "predictions for sample %s: %s and label %s "
    #                 % (sample_idx, mean, inference_labels[sample_idx])
    #             )

    #             print(
    #                 "mean and cov shape for this sample idx : ",
    #                 mean.shape,
    #                 cov_matr.shape,
    #             )
    #             print("mean and cov for this sample idx : ", mean, cov_matr)

    #             # PLot the 2D contour

    #             f, ax = plt.subplots(1, 2, figsize=(8, 3))

    #             # create  kernel
    #             m1 = mean[0]
    #             s1 = cov_matr
    #             k1 = multivariate_normal(mean=m1, cov=s1)

    #             # create a grid of (x,y) coordinates at which to evaluate the kernels
    #             xlim = (0, np.sqrt(len(inference_input[sample_idx])))
    #             ylim = (0, np.sqrt(len(inference_input[sample_idx])))
    #             xres = int(np.sqrt(len(inference_input[sample_idx])))
    #             yres = int(np.sqrt(len(inference_input[sample_idx])))

    #             x = np.linspace(xlim[0], xlim[1], xres)
    #             y = np.linspace(ylim[0], ylim[1], yres)
    #             xx, yy = np.meshgrid(x, y)

    #             # evaluate kernels at grid points
    #             xxyy = np.c_[xx.ravel(), yy.ravel()]
    #             zz = k1.pdf(xxyy)

    #             # reshape and plot image
    #             img = zz.reshape((xres, yres))
    #             ax[1].imshow(img)

    #             # show image with label
    #             print("this im tensor shape: ", inference_ims[sample_idx].shape)
    #             image_ex = inference_ims[sample_idx].cpu().detach().numpy()[0]
    #             image_label = inference_labels[sample_idx].cpu().detach().numpy()
    #             print("image ex shape ", image_ex.shape)
    #             ax[0].imshow(image_ex)

    #             rect1 = patches.Rectangle(
    #                 (int(image_label[0]), int(image_label[1])),
    #                 3,
    #                 3,
    #                 linewidth=2,
    #                 edgecolor="g",
    #                 facecolor="none",
    #             )
    #             ax[0].add_patch(rect1)

    #             rect2 = patches.Rectangle(
    #                 (int(image_label[0]), int(image_label[1])),
    #                 3,
    #                 3,
    #                 linewidth=2,
    #                 edgecolor="g",
    #                 facecolor="none",
    #             )
    #             ax[1].add_patch(rect2)
    #             rect3 = patches.Rectangle(
    #                 (int(m1[0]), int(m1[1])),
    #                 3,
    #                 3,
    #                 linewidth=2,
    #                 edgecolor="r",
    #                 facecolor="none",
    #             )
    #             ax[1].add_patch(rect3)

    #             plt.show()
    #             plt.close()

    #     # # This contains predictions for both tasks, flattened out
    #     # # The first half of the predictions is for the first task
    #     # # The second half is for the second task

    #     # # Plot training data as black stars
    #     # y1_ax.plot(self.all_training_input.detach().numpy(), self.all_training_labels[:, 0].detach().numpy(), 'k*')
    #     # # Predictive mean as blue line
    #     # y1_ax.plot(test_x.numpy(), mean[:, 0].numpy(), 'b')
    #     # # Shade in confidence
    #     # y1_ax.fill_between(test_x.numpy(), lower[:, 0].numpy(), upper[:, 0].numpy(), alpha=0.5)
    #     # y1_ax.set_ylim([-3, 3])
    #     # y1_ax.legend(['Observed Data', 'Mean', 'Confidence'])
    #     # y1_ax.set_title('Observed Values (Likelihood)')

    #     # # Plot training data as black stars
    #     # y2_ax.plot(train_x.detach().numpy(), train_y[:, 1].detach().numpy(), 'k*')
    #     # # Predictive mean as blue line
    #     # y2_ax.plot(test_x.numpy(), mean[:, 1].numpy(), 'b')
    #     # # Shade in confidence
    #     # y2_ax.fill_between(test_x.numpy(), lower[:, 1].numpy(), upper[:, 1].numpy(), alpha=0.5)
    #     # y2_ax.set_ylim([-3, 3])
    #     # y2_ax.legend(['Observed Data', 'Mean', 'Confidence'])
    #     # y2_ax.set_title('Observed Values (Likelihood)')

    def get_coords_from_heatmap(self, model_output, original_image_size):
        """Gets x,y coordinates from a model output. Here we use the final layer prediction of the U-Net,
            maybe resize and get coords as the peak pixel. Also return value of peak pixel.

        Args:
            output: model output - a stack of heatmaps

        Returns:
            [int, int]: predicted coordinates
        """
        prediction = model_output.mean
        lower, upper = model_output.confidence_region()

        cov_matr = model_output.covariance_matrix.cpu().detach().numpy()
        extra_info = {"lower": lower, "upper": upper, "cov_matr": cov_matr}
        return prediction,  extra_info

    def stitch_heatmap(self, patch_predictions, stitching_info, gauss_strength=0.5):
        """
        Use model outputs from a patchified image to stitch together a full resolution heatmap

        """
        orginal_im_size = [512, 512]
        full_heatmap = np.zeros((orginal_im_size[1], orginal_im_size[0]))
        patch_size_x = patch_predictions[0].shape[0]
        patch_size_y = patch_predictions[0].shape[1]

        for idx, patch in enumerate(patch_predictions):
            full_heatmap[
                stitching_info[idx][1]: stitching_info[idx][1] + patch_size_y,
                stitching_info[idx][0]: stitching_info[idx][0] + patch_size_x,
            ] += patch.detach.cpu().numpy()

        plt.imshow(full_heatmap)
        plt.show()

        raise NotImplementedError(
            "need to have original image size passed in because no longer assuming all have same size. see model base trainer for inspo"
        )
