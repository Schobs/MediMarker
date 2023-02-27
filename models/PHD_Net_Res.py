"""
PHDNet
(c) Schobs, Lawrence
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# from .preresnet import BasicBlock, Bottleneck


class PHD_Net_Res(nn.Module):
    """Implmentation of the PHDNet architecture [ref].
    Can use either the multi-branch scheme, heatmap branch only, or displacment branch only.

    """

    def __init__(self, branch_scheme):
        self.branch_scheme = branch_scheme
        super(PHD_Net_Res, self).__init__()

        self.layer1 = ConvGroup(1, 32)
        self.layer2 = ConvGroup(32, 64)
        self.layer3 = ConvGroup(64, 128)
        self.layer4 = ConvGroup(128, 256, pool=False)

        self.layer_reg = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, bias=True),
            BatchNorm(128),
            nn.GELU(),
            nn.Conv2d(128, 2, kernel_size=1, stride=1, bias=True)
        )

        self.layer_class = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, bias=True),
            BatchNorm(128),
            nn.GELU(),
            nn.Conv2d(128, 1, kernel_size=1, stride=1, bias=True))

    def forward(self, x):

        # print("the shape of x is:", x.shape)
        # out = []
        # print(x.shape)
        x = self.layer1(x)
        # print(x.shape)
        x = self.layer2(x)

        x = self.layer3(x)
        # print(x.shape)

        x = self.layer4(x)
        # print(x.shape)

        if self.branch_scheme == "multi" or "displacement":

            out_reg = self.layer_reg(x).unsqueeze(1)

        if self.branch_scheme == "multi" or "heatmap":

            out_class = self.layer_class(x)

        # out.append([out_class, out_reg])
        if self.branch_scheme == "multi":
            return [out_class, out_reg]
        elif self.branch_scheme == "heatmap":
            return out_class
        else:
            return out_reg


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-12, momentum=0.8, weight=False, bias=True):
        super().__init__(num_features, eps=eps, momentum=momentum)
        self.weight.data.fill_(1.0)
        self.bias.data.fill_(0.0)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias

# Allows us to set default arguments for the whole convolution itself.


class Conv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        default_conv_kwargs = {'kernel_size': 3, 'padding': 'same', 'bias': False}
        kwargs = {**default_conv_kwargs, **kwargs}
        super().__init__(*args, **kwargs)
        self.kwargs = kwargs

# can hack any changes to each residual group that you want directly in here


class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out, pool=True):
        super().__init__()
        self.pool = pool
        self.channels_in = channels_in
        self.channels_out = channels_out

        self.conv1 = Conv(channels_in, channels_out)
        self.pool1 = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out)
        self.activ = nn.GELU()

        self.conv2 = Conv(channels_out, channels_out)
        self.conv3 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out)
        self.norm3 = BatchNorm(channels_out)

    def forward(self, x):
        x = self.conv1(x)
        if self.pool:
            x = self.pool1(x)
        x = self.norm1(x)
        x = self.activ(x)

        residual = x

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.activ(x)
        x = x + residual

        return x
