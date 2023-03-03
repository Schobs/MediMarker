from torch import nn
from utils.setup.initialization import InitWeights_KaimingUniform
from losses.losses import (
    HeatmapLoss,
    IntermediateOutputLoss,
    AdaptiveWingLoss,
    SigmaLoss,
)
from models.TrUNet import TrUNet
import torch
import numpy as np

from utils.im_utils.heatmap_manipulation import get_coords, get_coords_fit_gauss
import matplotlib.pyplot as plt

# torch.multiprocessing.set_start_method('spawn')# good solution !!!!
from torchvision.transforms import Resize, InterpolationMode
from trainer.model_trainer_base import NetworkTrainer

from transforms.generate_labels import TrUNetLabelGenerator


class TrUNetTrainer(NetworkTrainer):
    """Class for the TrUNet trainer."""

    def __init__(self, **kwargs):

        super(TrUNetTrainer, self).__init__(**kwargs)

        # global config variable
        self.early_stop_patience = 250

        # Label generator
        self.train_label_generator = self.eval_label_generator = TrUNetLabelGenerator()

        # get model config parameters
        self.num_out_heatmaps = len(self.trainer_config.DATASET.LANDMARKS)
        self.base_num_features = self.trainer_config.MODEL.UNET.INIT_FEATURES
        self.min_feature_res = self.trainer_config.MODEL.UNET.MIN_FEATURE_RESOLUTION
        self.max_features = self.trainer_config.MODEL.UNET.MAX_FEATURES
        self.input_size = (
            self.trainer_config.SAMPLER.PATCH.SAMPLE_PATCH_SIZE
            if self.sampler_mode == "patch"
            else self.trainer_config.SAMPLER.INPUT_SIZE
        )

        # get arch config parameters
        self.num_resolution_layers = TrUNetTrainer.get_resolution_layers(
            self.input_size, self.min_feature_res
        )

        self.num_input_channels = 1
        self.conv_per_stage = 2
        self.conv_operation = nn.Conv2d
        self.dropout_operation = nn.Dropout2d
        self.normalization_operation = nn.InstanceNorm2d
        self.upsample_operation = nn.ConvTranspose2d
        self.norm_op_kwargs = {"eps": 1e-5, "affine": True}
        self.dropout_op_kwargs = {"p": 0, "inplace": True}  # don't do dropout
        self.activation_function = nn.LeakyReLU
        self.activation_kwargs = {"negative_slope": 1e-2, "inplace": True}
        self.pool_op_kernel_size = [(2, 2)] * (self.num_resolution_layers - 1)
        self.conv_op_kernel_size = [
            (3, 3)
        ] * self.num_resolution_layers  # remember set padding to (F-1)/2 i.e. 1
        self.conv_kwargs = {"stride": 1,
                            "dilation": 1, "bias": True, "padding": 1}

        # scheduler, initialiser and optimiser params
        self.weight_inititialiser = InitWeights_KaimingUniform(
            self.activation_kwargs["negative_slope"]
        )
        self.optimizer = torch.optim.SGD
        self.optimizer_kwargs = {
            "lr": self.initial_lr,
            "momentum": 0.99,
            "weight_decay": 3e-5,
            "nesterov": True,
        }

        # Deep supervision args
        self.deep_supervision = self.trainer_config.SOLVER.DEEP_SUPERVISION
        self.num_res_supervision = self.trainer_config.SOLVER.NUM_RES_SUPERVISIONS

        if not self.deep_supervision:
            self.num_res_supervision = 1  # just incase not set in config properly

        # Loss params
        loss_str = self.trainer_config.SOLVER.LOSS_FUNCTION
        if loss_str == "mse":
            self.individual_hm_loss = HeatmapLoss()
        elif loss_str == "awl":
            self.individual_hm_loss = AdaptiveWingLoss(
                hm_lambda_scale=self.trainer_config.MODEL.HM_LAMBDA_SCALE
            )
        else:
            raise ValueError(
                "the loss function %s is not implemented. Try mse or awl" % (
                    loss_str)
            )

        ################# Settings for saving checkpoints ##################################
        self.save_every = 25

    def initialize_network(self):
        """Initialise the network."""

        self.network = TrUNet(
            in_channels=self.num_input_channels,
            out_channels=1,
            img_size=512,
            feature_size=self.base_num_features,
            norm_name='batch',
            spatial_dims=2)
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
        """Initialise the optimiser and scheduler."""
        assert self.network is not None, "self.initialize_network must be called first"

        self.learnable_params = list(self.network.parameters())
        if self.regress_sigma:
            for sig in self.sigmas:
                self.learnable_params.append(sig)

        self.optimizer = self.optimizer(
            self.learnable_params, **self.optimizer_kwargs)

        print("Initialised optimizer.")

    def initialize_loss_function(self):
        """Initialise the loss function. Also initialise the deep supervision weights and loss. Potetntially initialise the sigma loss."""

        if self.deep_supervision:
            # first get weights for the layers. We don't care about the first two decoding levels
            # [::-1] because we don't use bottleneck layer. reverse bc the high res ones are important
            loss_weights = np.array(
                [1 / (2**i) for i in range(self.num_res_supervision)]
            )[::-1]
            loss_weights = loss_weights / loss_weights.sum()  # Normalise to add to 1
        else:
            loss_weights = [1]

        if self.regress_sigma:
            self.loss = IntermediateOutputLoss(
                self.individual_hm_loss,
                loss_weights,
                sigma_loss=True,
                sigma_weight=self.trainer_config.SOLVER.REGRESS_SIGMA_LOSS_WEIGHT,
            )
        else:
            self.loss = IntermediateOutputLoss(
                self.individual_hm_loss, loss_weights, sigma_loss=False
            )

        print("initialized Loss function.")

    def get_coords_from_heatmap(self, model_output, original_image_size):
        """Gets x,y coordinates from a model output. Here we use the final layer prediction of the U-Net,
            maybe resize and get coords as the peak pixel. Also return value of peak pixel.

        Args:
            model_output: model output - a stack of heatmaps

        Returns:
            [int, int]: predicted coordinates
        """

        extra_info = {"hm_max": []}

        # Get only the full resolution heatmap
        model_output = model_output[-1]

        final_heatmap = model_output
        original_image_size = original_image_size.cpu().detach().numpy()[
            :, ::-1, :]
        all_ims_same_size = np.all(
            original_image_size[0] == original_image_size)

        # Perform inference on each image
        input_size_coords, input_max_values = get_coords(model_output)

        # Save the predicted coordinates on the model-input size image
        extra_info["coords_og_size"] = input_size_coords

        # Depending on evaluation mode, we may need to resize the coords to the original image size
        if self.resize_first:

            # If all original images are the same size, we can resize as a batch, otherwise do them one by one.
            if all_ims_same_size:
                final_heatmap = Resize(
                    [original_image_size[0][0][0], original_image_size[0][1][0]],
                    interpolation=InterpolationMode.BICUBIC,
                )(final_heatmap)
                pred_coords, max_values = get_coords(final_heatmap)
            else:
                pred_coords = []
                max_values = []
                resized_heatmaps = []
                for im_idx, im_size in enumerate(original_image_size):
                    # print("this needs to be tested: model_trainer_unet.py  get_coords_from_heatmap when images are different sizes!")
                    resized_hm = Resize(
                        [im_size[0][0], im_size[1][0]],
                        interpolation=InterpolationMode.BICUBIC,
                    )(final_heatmap[im_idx])
                    # print("im idx, im_size, resized_hm shape ", im_idx, im_size, resized_hm.shape, torch.unsqueeze(resized_hm, 0).shape)
                    pc, mv = get_coords(torch.unsqueeze(resized_hm, 0))
                    # print("pc mv shapes ", pc.shape, mv.shape)
                    pred_coords.append(torch.squeeze(pc, 0))
                    max_values.append(torch.squeeze(mv, 0))
                    resized_heatmaps.append(resized_hm)
                pred_coords = torch.stack(pred_coords)
                max_values = torch.stack(max_values)
                final_heatmap = resized_heatmaps
        # If not resize, then just save the coords on the input size image
        else:
            pred_coords = input_size_coords
            max_values = input_max_values

        # Maybe fit a gaussian to the output heatmap and get the coords from that
        if self.fit_gauss_inference:
            pred_coords, max_values, fitted_dicts = get_coords_fit_gauss(
                final_heatmap, pred_coords, visualize=False
            )

        extra_info["hm_max"] = input_max_values
        extra_info["final_heatmaps"] = final_heatmap

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
                stitching_info[idx][1]: stitching_info[idx][1] + patch_size_y,
                stitching_info[idx][0]: stitching_info[idx][0] + patch_size_x,
            ] += patch.detach.cpu().numpy()

        plt.imshow(full_heatmap)
        plt.show()

        raise NotImplementedError(
            "need to have original image size passed in because no longer assuming all have same size. see model base trainer for inspo"
        )

    @staticmethod
    def get_resolution_layers(input_size, min_feature_res):
        """Defines the depth of the U-Net, depending on the input size and the minimum feature resolution.

        Args:
            input_size ([int, int]): Input image size to the network
            min_feature_res (_type_): Minimum feature resolution of the network

        Returns:
            counter (int): The number of resolution levels for the U-Net.
        """
        counter = 1
        while input_size[0] and input_size[1] >= min_feature_res * 2:
            counter += 1
            input_size = [x / 2 for x in input_size]
        return counter
