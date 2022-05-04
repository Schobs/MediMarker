'''
PHDNet
(c) Schobs, Lawrence
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

# from .preresnet import BasicBlock, Bottleneck




class PHDNet(nn.Module):

    def __init__(self, branch_scheme):
        self.branch_scheme = branch_scheme
        super(PHDNet, self).__init__()
     #   k_size = 3
        #padding = (kernal size -1)/2
        padding = 1
        run_stats = True
        momentum = 0.1
        # if is_train:
        #     print("train mode")
        #     run_stats = True
        #     momentum = 0.1
        # else:
        #     print("eval mode")
        #     run_stats = False
        #    momentum = 0.1

        self.layer1 =  nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, bias=True, padding=1),
            nn.BatchNorm2d(32, momentum=momentum, track_running_stats=run_stats),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        self.layer2 =  nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, bias=True, padding=1),
            nn.BatchNorm2d(32, momentum=momentum, track_running_stats=run_stats),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        self.layer3 =  nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, bias=True, padding=1),
            nn.BatchNorm2d(32, momentum=momentum, track_running_stats=run_stats),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        self.layer4 =  nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, bias=True, padding=1),
            nn.BatchNorm2d(32, momentum=momentum, track_running_stats=run_stats),
            nn.ReLU(inplace=True)
        )
        self.layer5 =  nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, bias=True, padding=1),
            nn.BatchNorm2d(32, momentum=momentum, track_running_stats=run_stats),
            nn.ReLU(inplace=True)
        )
        self.layer6 =  nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, bias=True, padding=1),
            nn.BatchNorm2d(32, momentum=momentum, track_running_stats=run_stats),
            nn.ReLU(inplace=True)
        )

        self.layer_reg = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=1, bias=True),
            nn.BatchNorm2d(64, momentum=momentum, track_running_stats=run_stats),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 96, kernel_size=1, stride=1, bias=True),
            nn.BatchNorm2d(96, momentum=momentum, track_running_stats=run_stats),
            nn.ReLU(inplace=True)
        )

        self.layer_class = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=1, bias=True),
            nn.BatchNorm2d(64, momentum=momentum, track_running_stats=run_stats),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 96, kernel_size=1, stride=1, bias=True),
            nn.BatchNorm2d(96, momentum=momentum, track_running_stats=run_stats),
            nn.ReLU(inplace=True) 
        )

        self.outReg = nn.Sequential(
            nn.Conv2d(96, 2, kernel_size=1, stride=1, bias=True)
           # nn.ReLU(inplace=True),
           #what acitivaton function for this??
        )

       
        self.outClass = nn.Sequential(
            nn.Conv2d(96, 1, kernel_size=1, stride=1, bias=True)
        )

    

        #dont apply sigmoid if using weighted loss as BCEwithlogits does sigmoid in it.


    def forward(self, x, sigmoid):

        # print("the shape of x is:", x.shape)
       # out = []
        # print(x.shape)
        x = self.layer1(x)
        # print(x.shape)
        x = self.layer2(x)
        # print(x.shape)

        x = self.layer3(x)
        # print(x.shape)

        x = self.layer4(x)
        # print(x.shape)

        x = self.layer5(x)
        # print(x.shape)

        x = self.layer6(x)
        # print(x.shape)

        if self.branch_scheme == 'patch_disp_gauss' or 'displacement_only':

            x_reg = self.layer_reg(x)
            # print("reg, ", x_reg.shape)
            out_reg = self.outReg(x_reg) 
            # print(out_reg.shape)

        if self.branch_scheme == 'patch_disp_gauss' or 'class_only':

            x_class = self.layer_class(x)
            # print("class, ", x_class.shape)

            
            out_class = self.outClass(x_class)
       



            if sigmoid == True:
                s = nn.Sigmoid()
                out_class = s(out_class)

      #  print(out_class.shape)

       # out.append([out_class, out_reg])
        if self.branch_scheme == 'patch_disp_gauss': 
            return [out_class, out_reg]
        elif self.branch_scheme == 'class_only':
            return out_class
        else: 
            return out_reg

        


