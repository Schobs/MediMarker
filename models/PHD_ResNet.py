import torch
import torch.nn as nn

#Based on 2021 paper from Julia Noothout,
#ref here: 


class PHDResNet(nn.Module):
    def __init__(self, block, layers, branch_scheme, num_landmarks):
        self.branch_scheme = branch_scheme
        self.num_landmarks = num_landmarks
        super(PHDResNet, self).__init__()

        run_stats = True
        momentum = 0.1
        self.in_channels = 32

        #convolutional layer with 16 (7 × 7) kernels and a stride of 2
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.in_channels, momentum=momentum, track_running_stats=run_stats),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.resnet1 = self.new_layer(block, 32, layers[0])

        self.resnet2 = self.new_layer(block, 64, layers[1], 2)

        self.resnet3 = self.new_layer(block, 128, layers[2])

        self.resnet4 = self.new_layer(block, 256, layers[3])

        #num landmarks * 2: get x, y for each landmark.
        self.layer_reg = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_landmarks * 2, kernel_size=1, stride=1, bias=True)
        )

        #get classification for each landmark 
        self.layer_class = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_landmarks, kernel_size=1, stride=1, bias=True))

    def forward(self, x):
      x = self.layer1(x)

      x = self.resnet1(x)

      x = self.resnet2(x)

      x = self.resnet3(x)
      
      x = self.resnet4(x)


      if self.branch_scheme == "multi" or "displacement":
        #make out_reg return the x, y
        out_reg = self.layer_reg(x).unsqueeze(1)

      if self.branch_scheme == "multi" or "heatmap":
        out_class = self.layer_class(x)

      if self.branch_scheme == "multi":
        return [out_class, out_reg]
      elif self.branch_scheme == "heatmap":
        return out_class
      else:
        return out_reg

    def new_layer(self, block, channels, blocks, stride=1):
        downsample = None
        layers = []

        if stride != 1 or self.in_channels != channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channels, 1, stride, bias=False),
                nn.BatchNorm2d(channels),
            )
        layers.append(block(self.in_channels, channels, stride, downsample))

        self.in_channels = channels

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, channels))

        return nn.Sequential(*layers)


class ResNetBlock(nn.Module):
    """
    Defines the ResNet block. n_pairs is the number of 
    convolutional layer pairs in the block
    """
    

    def __init__(self, in_channels, channels, stride=1, downsample=None):
        expansion = 1
        super().__init__()

        run_stats = True
        momentum = 0.1

        self.relu = nn.ReLU(inplace=True)

        self.layer1 = nn.Sequential(
          nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False),
          nn.BatchNorm2d(channels),
          nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
          nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
          nn.BatchNorm2d(channels),
        )

        self.downsample = downsample

    def forward(self, x):
        #use residual network shortcut idea
        shortcut = x

        out = self.layer1(x)
        out = self.layer2(out)

        #downsample defined with make layer function in link above
        if self.downsample is not None:
            shortcut = self.downsample(x)

        out += shortcut

        #apply ReLU after downsample
        out = self.relu(out)
        
        return out