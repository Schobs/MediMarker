from torch import nn
import os
from utils.setup.initialization import InitWeights_KaimingUniform
from losses.losses import MultiBranchPatchLoss
from models.PHD_Net import PHDNet
import torch
import numpy as np
from utils.im_utils.heatmap_manipulation import get_coords, candidate_smoothing
import matplotlib.pyplot as plt

# torch.multiprocessing.set_start_method('spawn')# good solution !!!!
from torchvision.transforms import Resize, InterpolationMode
from trainer.model_trainer_base import NetworkTrainer

from transforms.generate_labels import PHDNetLabelGenerator


class PHDNetTrainer(NetworkTrainer):
    """Class for the phdnet trainer stuff."""

    def __init__(self, **kwargs):

        super(PHDNetTrainer, self).__init__(**kwargs)

        # global config variable
        # self.trainer_config = trainer_config

        self.early_stop_patience = 1000

        # get model config parameters
        self.num_out_heatmaps = len(self.trainer_config.DATASET.LANDMARKS)
        self.base_num_features = self.trainer_config.MODEL.UNET.INIT_FEATURES
        self.min_feature_res = self.trainer_config.MODEL.UNET.MIN_FEATURE_RESOLUTION
        self.max_features = self.trainer_config.MODEL.UNET.MAX_FEATURES
        self.input_size = self.trainer_config.SAMPLER.INPUT_SIZE
        self.sample_patch_size = self.trainer_config.SAMPLER.PATCH.SAMPLE_PATCH_SIZE

        # get arch config parameters
        self.branch_scheme = self.trainer_config.MODEL.PHDNET.BRANCH_SCHEME
        self.maxpool_factor = self.trainer_config.MODEL.PHDNET.MAXPOOL_FACTOR
        self.class_label_scheme = self.trainer_config.MODEL.PHDNET.CLASS_LABEL_SCHEME

        # maxpool_factor, full_heatmap_resolution, class_label_scheme, sample_grid_size
        # Label generator
        self.train_label_generator = PHDNetLabelGenerator(
            self.maxpool_factor,
            self.training_resolution,
            self.class_label_scheme,
            self.sample_patch_size,
            self.trainer_config.MODEL.PHDNET.LOG_TRANSFORM_DISPLACEMENTS,
            clamp_dist=self.trainer_config.MODEL.PHDNET.CLAMP_DIST,
        )

        if self.trainer_config.SAMPLER.EVALUATION_SAMPLE_MODE == "full":
            eval_patch_size = self.training_resolution
        else:
            eval_patch_size = self.sample_patch_size

        self.eval_label_generator = PHDNetLabelGenerator(
            self.maxpool_factor,
            self.training_resolution,
            self.class_label_scheme,
            eval_patch_size,
            self.trainer_config.MODEL.PHDNET.LOG_TRANSFORM_DISPLACEMENTS,
            clamp_dist=self.trainer_config.MODEL.PHDNET.CLAMP_DIST,
        )

        # scheduler, initialiser and optimiser params
        self.weight_inititialiser = InitWeights_KaimingUniform()
        self.optimizer = torch.optim.SGD
        self.optimizer_kwargs = {
            "lr": self.initial_lr,
            "momentum": 0.99,
            "weight_decay": 3e-5,
            "nesterov": True,
        }

        ################# Settings for saving checkpoints ##################################
        self.save_every = 25

    def initialize_network(self):

        # Let's make the network
        self.network = PHDNet(self.branch_scheme)
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
        if self.regress_sigma:
            for sig in self.sigmas:
                self.learnable_params.append(sig)

        self.optimizer = self.optimizer(self.learnable_params, **self.optimizer_kwargs)

        print("Initialised optimizer.")

    def initialize_loss_function(self):

        # Loss params
        loss_str = self.trainer_config.SOLVER.LOSS_FUNCTION
        if loss_str == "mse":
            self.loss = MultiBranchPatchLoss(
                self.branch_scheme,
                self.class_label_scheme,
                self.trainer_config.MODEL.PHDNET.WEIGHT_DISP_LOSS_BY_HEATMAP,
            )
        else:
            raise ValueError(
                'the loss function %s is not implemented for PHD-Net. Try "mse".'
                % (loss_str)
            )

        print("initialized Loss function.")

    def get_coords_from_heatmap(self, model_output, original_image_size):
        """Gets x,y coordinates from a model output.
            maybe resize and get coords as the peak pixel. Also return value of peak pixel.

        Args:
            output: model output - a stack of heatmaps

        Returns:
            [int, int]: predicted coordinates
        """

        extra_info = {"hm_max": None}
        original_image_size = original_image_size.cpu().detach().numpy()[:, ::-1, :]

        model_output = [x.detach().cpu().numpy() for x in model_output]

        smoothed_candidate_maps = []

        for sample_idx in range(len(model_output[0])):

            sample_og_size = [
                original_image_size[sample_idx][0][0],
                original_image_size[sample_idx][1][0],
            ]
            # self.trainer_config.INFERENCE.DEBUG
            csm = candidate_smoothing(
                [model_output[0][sample_idx], model_output[1][sample_idx]],
                sample_og_size,
                self.maxpool_factor,
                log_displacement_bool=self.trainer_config.MODEL.PHDNET.LOG_TRANSFORM_DISPLACEMENTS,
                debug=self.trainer_config.INFERENCE.DEBUG,
            )
            if self.resize_first:
                csm = Resize(sample_og_size, interpolation=InterpolationMode.BICUBIC)(
                    csm
                )

            smoothed_candidate_maps.append(csm)

        smoothed_candidate_maps = torch.stack(smoothed_candidate_maps).to(self.device)
        # Get only the full resolution heatmap
        # model_output = model_output[-1]

        # final_heatmap = model_output
        # if self.resize_first:
        #     #torch resize does HxW so need to flip the diemsions
        #     final_heatmap = Resize(self.orginal_im_size[::-1], interpolation=  InterpolationMode.BICUBIC)(final_heatmap)

        # coords_from_uhm, arg_max_uhm = get_coords(torch.tensor(np.expand_dims(np.expand_dims(upscaled_hm, axis=0), axis=0)))

        pred_coords, max_values = get_coords(smoothed_candidate_maps)
        extra_info["hm_max"] = max_values
        extra_info["final_heatmaps"] = smoothed_candidate_maps

        extra_info["debug_candidate_smoothed_maps"] = smoothed_candidate_maps

        return pred_coords, extra_info

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
                stitching_info[idx][1] : stitching_info[idx][1] + patch_size_y,
                stitching_info[idx][0] : stitching_info[idx][0] + patch_size_x,
            ] += patch.detach.cpu().numpy()

        plt.imshow(full_heatmap)
        plt.show()

        raise NotImplementedError(
            "need to have original image size passed in because no longer assuming all have same size. see model base trainer for inspo"
        )

    # def set_training_dataloaders(self):
    #     super(PHDNetTrainer, self).set_training_dataloaders()
