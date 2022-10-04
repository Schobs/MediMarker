
from torch import nn
import os
from utils.setup.initialization import InitWeights_KaimingUniform
from losses import HeatmapLoss, IntermediateOutputLoss, AdaptiveWingLoss, SigmaLoss
from models.UNet_Classic import UNet
from visualisation import visualize_heat_pred_coords
import torch
import numpy as np
from time import time
# from dataset import ASPIRELandmarks
from datasets.dataset import DatasetBase
# import multiprocessing as mp
import ctypes
import copy
from torch.utils.data import DataLoader
from utils.im_utils.heatmap_manipulation import get_coords, get_coords_fit_gauss
from torch.cuda.amp import GradScaler, autocast
import imgaug
import torch.multiprocessing as mp
from torch.multiprocessing import Pool, Process, set_start_method
import matplotlib.pyplot as plt

# torch.multiprocessing.set_start_method('spawn')# good solution !!!!
from torchvision.transforms import Resize,InterpolationMode
from trainer.model_trainer_base import NetworkTrainer

from transforms.generate_labels import UNetLabelGenerator
class UnetTrainer(NetworkTrainer):
    """ Class for the u-net trainer stuff.
    """

    def __init__(self, trainer_config= None, is_train=True, output_folder=None, comet_logger=None, profiler=None):


        super(UnetTrainer, self).__init__(trainer_config, is_train, output_folder, comet_logger, profiler)

      
        #global config variable
        self.trainer_config = trainer_config
        self.early_stop_patience = 250


        #Label generator
        self.train_label_generator = self.eval_label_generator = UNetLabelGenerator()
   
        #get model config parameters
        self.num_out_heatmaps = len(self.trainer_config.DATASET.LANDMARKS)
        self.base_num_features = self.trainer_config.MODEL.UNET.INIT_FEATURES
        self.min_feature_res = self.trainer_config.MODEL.UNET.MIN_FEATURE_RESOLUTION
        self.max_features = self.trainer_config.MODEL.UNET.MAX_FEATURES
        self.input_size = self.trainer_config.SAMPLER.INPUT_SIZE
        self.orginal_im_size = self.trainer_config.DATASET.ORIGINAL_IMAGE_SIZE


        #get arch config parameters
        self.num_resolution_layers = UnetTrainer.get_resolution_layers(self.input_size,  self.min_feature_res)

        self.num_input_channels = 1
        self.conv_per_stage = 2
        self.conv_operation = nn.Conv2d
        self.dropout_operation = nn.Dropout2d
        self.normalization_operation = nn.InstanceNorm2d
        self.upsample_operation = nn.ConvTranspose2d
        self.norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        self.dropout_op_kwargs = {'p': 0, 'inplace': True} # don't do dropout
        self.activation_function =  nn.LeakyReLU
        self.activation_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.pool_op_kernel_size = [(2,2)] * (self.num_resolution_layers -1)
        self.conv_op_kernel_size = [(3,3)] * self.num_resolution_layers # remember set padding to (F-1)/2 i.e. 1
        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True, 'padding': 1}

        #scheduler, initialiser and optimiser params
        self.weight_inititialiser = InitWeights_KaimingUniform(self.activation_kwargs['negative_slope'])
        self.optimizer= torch.optim.SGD
        self.optimizer_kwargs =  {"lr": self.initial_lr, "momentum": 0.99, "weight_decay": 3e-5, "nesterov": True}

        #Deep supervision args
        self.deep_supervision= self.trainer_config.SOLVER.DEEP_SUPERVISION
        self.num_res_supervision = self.trainer_config.SOLVER.NUM_RES_SUPERVISIONS

        if not self.deep_supervision:
            self.num_res_supervision = 1 #just incase not set in config properly


        #Loss params
        loss_str = self.trainer_config.SOLVER.LOSS_FUNCTION
        if loss_str == "mse":
            self.individual_hm_loss = HeatmapLoss()
        elif loss_str =="awl":
            self.individual_hm_loss = AdaptiveWingLoss(hm_lambda_scale=self.model_config.MODEL.HM_LAMBDA_SCALE)
        else:
            raise ValueError("the loss function %s is not implemented. Try mse or awl" % (loss_str))



        ################# Settings for saving checkpoints ##################################
        self.save_every = 25






    def initialize_network(self):

        # Let's make the network
        self.network = UNet(input_channels=self.num_input_channels, base_num_features=self.base_num_features, num_out_heatmaps=self.num_out_heatmaps,
            num_resolution_levels= self.num_resolution_layers, conv_operation=self.conv_operation, normalization_operation=self.normalization_operation,
            normalization_operation_config=self.norm_op_kwargs, activation_function= self.activation_function, activation_func_config= self.activation_kwargs,
            weight_initialization=self.weight_inititialiser, strided_convolution_kernels = self.pool_op_kernel_size, convolution_kernels= self.conv_op_kernel_size,
            convolution_config=self.conv_kwargs, upsample_operation=self.upsample_operation, max_features=self.max_features, deep_supervision=self.deep_supervision
        
        )
        self.network.to(self.device)

        #Log network and initial weights
        if self.comet_logger:
            self.comet_logger.set_model_graph(str(self.network))
            print("Logged the model graph.")

     
        print("Initialized network architecture. #parameters: ", sum(p.numel() for p in self.network .parameters()))

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"

      
        self.learnable_params = list(self.network.parameters())
        if self.regress_sigma:
            for sig in self.sigmas:
                self.learnable_params.append(sig)

        self.optimizer = self.optimizer(self.learnable_params, **self.optimizer_kwargs)

   
        print("Initialised optimizer.")


    def initialize_loss_function(self):

        if self.deep_supervision:
            #first get weights for the layers. We don't care about the first two decoding levels
            #[::-1] because we don't use bottleneck layer. reverse bc the high res ones are important
            loss_weights = np.array([1 / (2 ** i) for i in range(self.num_res_supervision)])[::-1] 
            loss_weights = (loss_weights / loss_weights.sum()) #Normalise to add to 1
        else:
            loss_weights = [1]

        if self.regress_sigma:
            self.loss = IntermediateOutputLoss(self.individual_hm_loss, loss_weights,sigma_loss=True, sigma_weight=self.trainer_config.SOLVER.REGRESS_SIGMA_LOSS_WEIGHT )
        else:
            self.loss = IntermediateOutputLoss(self.individual_hm_loss, loss_weights,sigma_loss=False )

        print("initialized Loss function.")



    def get_coords_from_heatmap(self, output):
        """ Gets x,y coordinates from a model output. Here we use the final layer prediction of the U-Net,
            maybe resize and get coords as the peak pixel. Also return value of peak pixel.

        Args:
            output: model output - a stack of heatmaps

        Returns:
            [int, int]: predicted coordinates
        """

        extra_info = {"hm_max": []}

        #Get only the full resolution heatmap
        output = output[-1]

        final_heatmap = output

    #  get_coords_fit_gauss(images, predicted_coords_all, visualize=False)
        input_size_coords, input_max_values = get_coords(output)
        # if self.fit_gauss_inference:
        #     input_size_coords, input_max_values, input_fitted_dicts = get_coords_fit_gauss(output, input_size_coords, visualize=True)
        # else:
        # print("input_size_coords", input_size_coords[0].shape, input_size_coords[1].shape)
        # extra_info["coords_og_size"] = [[input_size_coords[0][idx], input_size_coords[1][idx]] for idx in range(len(input_size_coords[0]))]
        extra_info["coords_og_size"] = input_size_coords
        if self.resize_first:
            #torch resize does HxW so need to flip the diemsions
            final_heatmap = Resize(self.orginal_im_size[::-1], interpolation=  InterpolationMode.BICUBIC)(final_heatmap)
        

        pred_coords, max_values = get_coords(final_heatmap)
        if self.fit_gauss_inference:
            pred_coords, max_values, fitted_dicts = get_coords_fit_gauss(final_heatmap, pred_coords, visualize=False)
        # else:
        extra_info["hm_max"] = (max_values)

        del final_heatmap
  
        return pred_coords, extra_info


    
    
    def stitch_heatmap(self, patch_predictions, stitching_info, gauss_strength=0.5):
        '''
        Use model outputs from a patchified image to stitch together a full resolution heatmap
        
        '''


        full_heatmap = np.zeros((self.orginal_im_size[1], self.orginal_im_size[0]))
        patch_size_x = patch_predictions[0].shape[0]
        patch_size_y = patch_predictions[0].shape[1]

        for idx, patch in enumerate(patch_predictions):
            full_heatmap[stitching_info[idx][1]:stitching_info[idx][1]+patch_size_y, stitching_info[idx][0]:stitching_info[idx][0]+patch_size_x] += patch.detach.cpu().numpy()

        plt.imshow(full_heatmap)
        plt.show()



        

    # def predict_heatmaps_and_coordinates(self, data_dict,  return_all_layers = False, resize_to_og=False,):
    #     data =(data_dict['image']).to( self.device )
    #     target = [x.to(self.device) for x in data_dict['label']]
    #     from_which_level_supervision = self.num_res_supervision 

    #     if self.deep_supervision:
    #         output = self.network(data)[-from_which_level_supervision:]
    #     else:
    #         output = self.network(data)

        
    #     l, loss_dict = self.loss(output, target, self.sigmas)

    #     final_heatmap = output[-1]
    #     if resize_to_og:
    #         #torch resize does HxW so need to flip the dimesions for resize
    #         final_heatmap = Resize(self.orginal_im_size[::-1], interpolation=  InterpolationMode.BICUBIC)(final_heatmap)

    #     predicted_coords, max_values = get_coords(final_heatmap)

    #     heatmaps_return = output
    #     if not return_all_layers:
    #         heatmaps_return = output[-1] #only want final layer


    #     return heatmaps_return, final_heatmap, predicted_coords, l.detach().cpu().numpy()

    
    


    @staticmethod
    def get_resolution_layers(input_size,  min_feature_res):
        counter=1
        while input_size[0] and input_size[1] >= min_feature_res*2:
            counter+=1
            input_size = [x/2 for x in input_size]
        return counter






