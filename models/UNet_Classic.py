# =============================================================================
# Author: Lawrence Schobs, laschobs@sheffield.ac.uk
# =============================================================================

"""
Define the learning model and configure training parameters.
"""

from copy import deepcopy
import re


import torch.nn as nn
from torch import cat as torch_concat


class ConvNormNonlin(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        conv_op=nn.Conv2d,
        conv_kwargs=None,
        norm_op=nn.InstanceNorm2d,
        norm_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs=None,
    ):
        super(ConvNormNonlin, self).__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {"negative_slope": 1e-2, "inplace": True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {"eps": 1e-5, "affine": True, "momentum": 0.1}
        if conv_kwargs is None:
            conv_kwargs = {
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "dilation": 1,
                "bias": True,
            }

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        # Convolutional operation
        self.convolution = self.conv_op(
            input_channels, output_channels, **self.conv_kwargs
        )

        # Normalization operation
        self.normalization = self.norm_op(output_channels, **self.norm_op_kwargs)

        # Activation function
        self.activation = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        x = self.convolution(x)
        return self.activation(self.normalization(x))


class UNet(nn.Module):
    """Regular classifier network for video input.

    Args:
        input_size (int, optional): the dimension of the final feature vector. Defaults to 512.
        n_channel (int, optional): the number of channel for Linear and BN layers.
        dropout_keep_prob (int, optional): the dropout probability for keeping the parameters.
        n_class (int, optional): the number of classes. Defaults to 8.
    """

    def __init__(
        self,
        input_channels,
        base_num_features,
        num_out_heatmaps,
        num_resolution_levels,
        conv_operation,
        normalization_operation,
        normalization_operation_config,
        activation_function,
        activation_func_config,
        weight_initialization,
        strided_convolution_kernels,
        convolution_kernels,
        convolution_config,
        upsample_operation,
        deep_supervision,
        max_features=512,
    ):

        super(UNet, self).__init__()

        self.input_channels = input_channels
        self.base_num_features = base_num_features
        self.num_out_heatmaps = num_out_heatmaps
        self.num_resolution_levels = num_resolution_levels
        self.conv_operation = conv_operation
        self.normalization_operation = normalization_operation
        self.normalization_operation_config = normalization_operation_config
        self.activation_function = activation_function
        self.activation_func_config = activation_func_config
        self.weight_initialization = weight_initialization
        self.strided_convolution_kernels = strided_convolution_kernels
        self.convolution_kernels = convolution_kernels
        self.convolution_config = convolution_config
        self.upsample_operation = upsample_operation
        self.max_features = max_features
        self.deep_supervision = deep_supervision

        # Define the network
        self.conv_blocks_encoder = []
        self.upsample_transp_convs = []
        self.conv_blocks_decoder = []
        self.intermediate_outputs = []

        # set initial number of kernels and input channels
        output_features = self.base_num_features
        input_features = self.input_channels

        # The Encoder part
        for res_level in range(self.num_resolution_levels):

            # Get the convoltion kernels for this resolution level. We use 2 conv-norm-activation sequences per level
            first_conv_kwargs = deepcopy(self.convolution_config)
            first_conv_kwargs["kernel_size"] = self.convolution_kernels[res_level]

            second_conv_kwargs = deepcopy(self.convolution_config)
            second_conv_kwargs["kernel_size"] = self.convolution_kernels[res_level]

            # If the first convolution, do not downsample, otherwise downsample using strided convolutions
            if res_level == 0:
                first_stride = 1
            else:
                first_stride = strided_convolution_kernels[
                    res_level - 1
                ]  # -1 because the first res_level didn't use

            first_conv_kwargs["stride"] = first_stride
            second_conv_kwargs["stride"] = 1

            self.conv_blocks_encoder.append(
                nn.Sequential(
                    ConvNormNonlin(
                        input_features,
                        output_features,
                        self.conv_operation,
                        first_conv_kwargs,
                        self.normalization_operation,
                        self.normalization_operation_config,
                        self.activation_function,
                        self.activation_func_config,
                    ),
                    ConvNormNonlin(
                        output_features,
                        output_features,
                        self.conv_operation,
                        second_conv_kwargs,
                        self.normalization_operation,
                        self.normalization_operation_config,
                        self.activation_function,
                        self.activation_func_config,
                    ),
                )
            )

            input_features = output_features
            output_features = min(
                int(output_features * 2), self.max_features
            )  # We double the features each level to a preset maximum (default: 512)

        # The Decoder part. We need to transpose the convolution to increase resolution. During training
        # we concatonate these features with a skip connection to mirror the same number of features. Then we need to
        # apply the classic block of 2 conv-norm-activation blocks

        # initialise feature sizes
        nfeatures_from_encoder = output_features
        for res_level in range(
            self.num_resolution_levels - 1
        ):  # -1 because the encoder block covered the lowest resolution already

            # print("self conv_blovks encoder ", -(1 + res_level), " is ", self.conv_blocks_encoder[-(1 + res_level)][-1].output_channels)
            nfeatures_from_encoder = self.conv_blocks_encoder[-(1 + res_level)][
                -1
            ].output_channels  # prev level #features

            nfeatures_from_skip = self.conv_blocks_encoder[-(2 + res_level)][
                -1
            ].output_channels  # target #features
            n_features_after_tu_and_concat = nfeatures_from_skip * 2

            # Transpose Convolution to upsample resolution
            self.upsample_transp_convs.append(
                self.upsample_operation(
                    nfeatures_from_encoder,
                    nfeatures_from_skip,
                    strided_convolution_kernels[-(res_level + 1)],
                    strided_convolution_kernels[-(res_level + 1)],
                    bias=False,
                )
            )

            # Now we know the input will have same # features as skip connection, and doubled resolution to also match skip.

            # Get the convolution kernels for this resolution level. We use 2 conv-norm-activation sequences per level.
            # Since we are doing no pooling use the convs, they both have stride 1.
            conv_kwargs = deepcopy(self.convolution_config)
            conv_kwargs["kernel_size"] = self.convolution_kernels[-(res_level + 1)]
            conv_kwargs["stride"] = 1

            self.conv_blocks_decoder.append(
                nn.Sequential(
                    ConvNormNonlin(
                        n_features_after_tu_and_concat,
                        nfeatures_from_skip,
                        self.conv_operation,
                        conv_kwargs,
                        self.normalization_operation,
                        self.normalization_operation_config,
                        self.activation_function,
                        self.activation_func_config,
                    ),
                    ConvNormNonlin(
                        nfeatures_from_skip,
                        nfeatures_from_skip,
                        self.conv_operation,
                        conv_kwargs,
                        self.normalization_operation,
                        self.normalization_operation_config,
                        self.activation_function,
                        self.activation_func_config,
                    ),
                )
            )

        # Now do intermediate decoder levels for deep supervision
        # Get the very last layer of each decoder block, and apply 1x1 convolution for output heatmaps.

        for deep_sup in range(len(self.conv_blocks_decoder)):
            self.intermediate_outputs.append(
                self.conv_operation(
                    self.conv_blocks_decoder[deep_sup][-1].output_channels,
                    self.num_out_heatmaps,
                    1,
                    1,
                    0,
                    1,
                    1,
                )
            )

        # Initialise blocks as module lists

        # Define the network
        self.conv_blocks_encoder = nn.ModuleList(self.conv_blocks_encoder)
        self.upsample_transp_convs = nn.ModuleList(self.upsample_transp_convs)
        self.conv_blocks_decoder = nn.ModuleList(self.conv_blocks_decoder)
        self.intermediate_outputs = nn.ModuleList(self.intermediate_outputs)

        # Initialise weights
        self.apply(self.weight_initialization)

    def forward(self, x):
        skips = []
        seg_outputs = []

        # Encoder
        for encoder_lvl in range(len(self.conv_blocks_encoder) - 1):
            x = self.conv_blocks_encoder[encoder_lvl](x)

            skips.append(x)

        # Bottleneck (don't add to skip)
        x = self.conv_blocks_encoder[-1](x)

        # Decoder (with transpose convolutions)
        for decoder_lvl in range(len(self.conv_blocks_decoder)):

            x = self.upsample_transp_convs[decoder_lvl](x)

            x = torch_concat((x, skips[-(decoder_lvl + 1)]), dim=1)

            x = self.conv_blocks_decoder[decoder_lvl](x)

            if self.deep_supervision:
                seg_outputs.append(
                    (self.intermediate_outputs[decoder_lvl](x))
                )  # Don't use an activation function
        if not self.deep_supervision:
            seg_outputs.append(
                (self.intermediate_outputs[-1](x))
            )  # if not deep supervision, only use the final layer

        return seg_outputs

    def get_intermediate_representation(self, x, layer):
        """
        This function obtains the output of a specified layer in the UNet architecture when given an input. The specified
        layer could be an encoder layer (denoted by "E" followed by the layer number, e.g., "E1" for the first encoder 
        layer), a decoder layer (denoted by "D" followed by the layer number), or the bottleneck layer (denoted by "B").

        Args:
            x (torch.Tensor): The input tensor to the model.
            layer (str): The layer for which the output is required. The layer is specified as a string with the following 
                        format: "EX" for encoder layers (where X is the layer number), "DX" for decoder layers, and "B" for 
                        the bottleneck layer.

        Returns:
            torch.Tensor: The output tensor from the specified layer.

        Raises:
            Exception: An exception is raised if the layer specified does not exist in the model.
            ValueError: A ValueError is raised if the layer parameter is not a string or does not follow the required format.
        """

        # Check that the layer parameter is in the correct format
        if not re.match(r'^(E|D)\d+$|^B$', layer):
            raise ValueError(
                f"Invalid layer parameter {layer}. Layer parameter should start with 'E' or 'D', followed by a number, or be 'B'. Try E1, E2, B, D1, D2 etc.")

        skips = []
        if layer.startswith("E"):
            try:
                layer_idx = int(layer[1:])
                for i in range(layer_idx):
                    x = self.conv_blocks_encoder[i](x)
                return x
            except IndexError:
                raise Exception(f"No encoder layer exists at index: {layer_idx}")
        elif layer.startswith("D"):
            try:
                layer_idx = int(layer[1:])
                for i in range(len(self.conv_blocks_encoder) - 1):
                    x = self.conv_blocks_encoder[i](x)
                    skips.append(x)

                # Bottleneck (don't add to skip)
                x = self.conv_blocks_encoder[-1](x)

                for decoder_lvl in range(layer_idx):
                    x = self.upsample_transp_convs[decoder_lvl](x)
                    x = torch_concat((x, skips[-(decoder_lvl + 1)]), dim=1)
                    x = self.conv_blocks_decoder[decoder_lvl](x)
                return x
            except IndexError:
                raise Exception(f"No decoder layer exists at index: {layer_idx}")
        elif layer == "B":
            try:
                # -1 because we don't want to add the last encoder layer to skips
                for encoder_lvl in range(len(self.conv_blocks_encoder) - 1):
                    x = self.conv_blocks_encoder[encoder_lvl](x)
                x = self.conv_blocks_encoder[-1](x)
                return x
            except IndexError:
                raise Exception("No bottleneck layer exists.")
        else:
            raise ValueError(
                "Invalid layer parameter. Layer parameter should start with 'E' or 'D', or be 'B'. Try E1, E2, B, D1, D2 etc.")
