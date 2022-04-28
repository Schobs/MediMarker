
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
from dataset import ASPIRELandmarks
# import multiprocessing as mp
import ctypes
import copy
from torch.utils.data import DataLoader
from utils.im_utils.heatmap_manipulation import get_coords
from torch.cuda.amp import GradScaler, autocast
import imgaug
import torch.multiprocessing as mp
from torch.multiprocessing import Pool, Process, set_start_method

# torch.multiprocessing.set_start_method('spawn')# good solution !!!!
from torchvision.transforms import Resize,InterpolationMode
from trainer.model_trainer_base import NetworkTrainer

class UnetTrainer(NetworkTrainer):
    """ Class for the u-net trainer stuff.
    """

    def __init__(self, model_config= None, output_folder=None, logger=None, profiler=None):


        super(UnetTrainer, self).__init__()

        #Device

        #early stopping
        self.early_stop_patience = 150

        #global config variable
        self.model_config = model_config

        #Trainer variables
        self.perform_validation = model_config.TRAINER.PERFORM_VALIDATION

        self.continue_checkpoint = model_config.MODEL.CHECKPOINT
        self.logger = logger
        self.profiler = profiler
        self.verbose_logging = model_config.OUTPUT.VERBOSE
        self.auto_mixed_precision = model_config.SOLVER.AUTO_MIXED_PRECISION
        self.fold= model_config.TRAINER.FOLD
        self.output_folder = output_folder

      
        #Dataloader info
        self.data_loader_batch_size = model_config.SOLVER.DATA_LOADER_BATCH_SIZE
        self.num_val_batches_per_epoch = 50
        self.num_batches_per_epoch = model_config.SOLVER.MINI_BATCH_SIZE

        #Training params
        self.max_num_epochs =  model_config.SOLVER.MAX_EPOCHS
        self.initial_lr = 1e-2

        #Sigma for Gaussian heatmaps
        self.regress_sigma = model_config.SOLVER.REGRESS_SIGMA
        self.sigmas = [torch.tensor(x, dtype=float, device=self.device, requires_grad=True) for x in np.repeat(self.model_config.MODEL.GAUSS_SIGMA, len(model_config.DATASET.LANDMARKS))]

        
        #get validaiton params
        if model_config.INFERENCE.EVALUATION_MODE == 'use_input_size' or  model_config.DATASET.ORIGINAL_IMAGE_SIZE == model_config.DATASET.INPUT_SIZE:
            self.use_full_res_coords =False
            self.resize_first = False
        elif model_config.INFERENCE.EVALUATION_MODE == 'scale_heatmap_first':
            self.use_full_res_coords =True
            self.resize_first = True
        elif model_config.INFERENCE.EVALUATION_MODE == 'scale_pred_coords':
            self.use_full_res_coords =True
            self.resize_first = False
        else:
            raise ValueError("value for cg.INFERENCE.EVALUATION_MODE not recognised. Choose from: scale_heatmap_first, scale_pred_coords, use_input_size")

      

        #get model config parameters
        self.num_out_heatmaps = len(model_config.DATASET.LANDMARKS)
        self.base_num_features = model_config.MODEL.UNET.INIT_FEATURES
        self.min_feature_res = model_config.MODEL.UNET.MIN_FEATURE_RESOLUTION
        self.max_features = model_config.MODEL.UNET.MAX_FEATURES
        self.input_size = model_config.DATASET.INPUT_SIZE
        self.orginal_im_size = model_config.DATASET.ORIGINAL_IMAGE_SIZE


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
        self.deep_supervision= model_config.SOLVER.DEEP_SUPERVISION
        self.num_res_supervision = model_config.SOLVER.NUM_RES_SUPERVISIONS

        if not self.deep_supervision:
            self.num_res_supervision = 1 #just incase not set in config properly


        #Loss params
        loss_str = model_config.SOLVER.LOSS_FUNCTION
        if loss_str == "mse":
            self.individual_hm_loss = HeatmapLoss()
        elif loss_str =="awl":
            self.individual_hm_loss = AdaptiveWingLoss(hm_lambda_scale=self.model_config.MODEL.HM_LAMBDA_SCALE)
        else:
            raise ValueError("the loss function %s is not implemented. Try mse or awl" % (loss_str))

      

        ################# Settings for saving checkpoints ##################################
        self.save_every = 25
        self.save_latest_only = model_config.TRAINER.SAVE_LATEST_ONLY  # if false it will not store/overwrite _latest but separate files each




    def initialize(self, training_bool=True):

        super(UnetTrainer, self).initialize()



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
        if self.logger:
            self.logger.set_model_graph(str(self.network))
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
            self.loss = IntermediateOutputLoss(self.individual_hm_loss, loss_weights,sigma_loss=True, sigma_weight=self.model_config.SOLVER.REGRESS_SIGMA_LOSS_WEIGHT )
        else:
            self.loss = IntermediateOutputLoss(self.individual_hm_loss, loss_weights,sigma_loss=False )

   

        print("initialized Loss function.")

    def maybe_update_lr(self, epoch=None, exponent=0.9):

        super(UnetTrainer, self).maybe_update_lr(epoch, exponent)

       

    def _maybe_init_amp(self):
        super(UnetTrainer, self)._maybe_init_amp()

   

    def train(self):
        super(UnetTrainer, self).train()

    
    def get_coords_from_model_output(self, output):
        """ Gets x,y coordinates from a model output. Here we use the final layer prediction of the U-Net,
            maybe resize and get coords as the peak pixel.

        Args:
            output: model output - a stack of heatmaps

        Returns:
            [int, int]: predicted coordinates
        """

        final_heatmap = output[-1]
        if self.resize_first:
            #torch resize does HxW so need to flip the diemsions
            final_heatmap = Resize(self.orginal_im_size[::-1], interpolation=  InterpolationMode.BICUBIC)(final_heatmap)
        pred_coords = get_coords(final_heatmap)
        del final_heatmap

        return pred_coords




    def predict_heatmaps_and_coordinates(self, data_dict,  return_all_layers = False, resize_to_og=False,):
        data =(data_dict['image']).to( self.device )
        target = [x.to(self.device) for x in data_dict['label']]
        from_which_level_supervision = self.num_res_supervision 

        if self.deep_supervision:
            output = self.network(data)[-from_which_level_supervision:]
        else:
            output = self.network(data)

        
        l = self.loss(output, target, self.sigmas)

        final_heatmap = output[-1]
        if resize_to_og:
            #torch resize does HxW so need to flip the dimesions for resize
            final_heatmap = Resize(self.orginal_im_size[::-1], interpolation=  InterpolationMode.BICUBIC)(final_heatmap)

        predicted_coords = get_coords(final_heatmap)

        heatmaps_return = output
        if not return_all_layers:
            heatmaps_return = output[-1] #only want final layer


        return heatmaps_return, final_heatmap, predicted_coords, l.detach().cpu().numpy()


    def run_iteration(self, generator, dataloader, backprop, get_coord_error=False, coord_error_list=None):
        return super(UnetTrainer, self).run_iteration(generator, dataloader, backprop, get_coord_error, coord_error_list)

    

    def set_training_dataloaders(self):

        np_sigmas = [x.cpu().detach().numpy() for x in self.sigmas]
    
        train_dataset = ASPIRELandmarks(
            annotation_path =self.model_config.DATASET.SRC_TARGETS,
            landmarks = self.model_config.DATASET.LANDMARKS,
            split = "training",
            root_path = self.model_config.DATASET.ROOT,
            sigmas =  np_sigmas,
            cv = self.model_config.TRAINER.FOLD,
            cache_data = self.model_config.TRAINER.CACHE_DATA,
            normalize=True,
            num_res_supervisions = self.num_res_supervision,
            debug=self.model_config.DATASET.DEBUG ,
            original_image_size= self.model_config.DATASET.ORIGINAL_IMAGE_SIZE,
            input_size =  self.model_config.DATASET.INPUT_SIZE,
            hm_lambda_scale = self.model_config.MODEL.HM_LAMBDA_SCALE,
            data_augmentation_strategy = self.model_config.DATASET.DATA_AUG,
            data_augmentation_package = self.model_config.DATASET.DATA_AUG_PACKAGE,
            regress_sigma = self.model_config.SOLVER.REGRESS_SIGMA

 
        )


        if self.perform_validation:
            val_split= "validation"
            
            valid_dataset = ASPIRELandmarks(
                annotation_path =self.model_config.DATASET.SRC_TARGETS,
                landmarks = self.model_config.DATASET.LANDMARKS,
                split = val_split,
                root_path = self.model_config.DATASET.ROOT,
                sigmas =  np_sigmas,
                cv = self.model_config.TRAINER.FOLD,
                cache_data = self.model_config.TRAINER.CACHE_DATA,
                normalize=True,
                num_res_supervisions = self.num_res_supervision,
                debug=self.model_config.DATASET.DEBUG,
                data_augmentation_strategy =None,
                original_image_size= self.model_config.DATASET.ORIGINAL_IMAGE_SIZE,
                input_size =  self.model_config.DATASET.INPUT_SIZE,
                hm_lambda_scale = self.model_config.MODEL.HM_LAMBDA_SCALE,
                regress_sigma = self.model_config.SOLVER.REGRESS_SIGMA

            )
        else:
            val_split = "training"
            valid_dataset = train_dataset
            print("WARNING: NOT performing validation. Instead performing \"validation\" on training set for coord error metrics.")



        #If debug use only main thread to load data bc we only want to show a single plot on screen.
        #If num_workers=0 we are only using the main thread, so persist_workers = False.
        if self.model_config.DATASET.DEBUG or self.model_config.DATASET.NUM_WORKERS == 0:
            persist_workers = False
            num_workers_cfg=0
        else:
            persist_workers = True
            num_workers_cfg= self.model_config.DATASET.NUM_WORKERS    
       

        self.train_dataloader = DataLoader(train_dataset, batch_size=self.model_config.SOLVER.DATA_LOADER_BATCH_SIZE, shuffle=True, num_workers=self.model_config.DATASET.NUM_WORKERS, persistent_workers=persist_workers, worker_init_fn=UnetTrainer.worker_init_fn, pin_memory=True )
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=self.model_config.SOLVER.DATA_LOADER_BATCH_SIZE, shuffle=False, num_workers=num_workers_cfg, persistent_workers=persist_workers, worker_init_fn=UnetTrainer.worker_init_fn, pin_memory=True )

    @staticmethod
    def get_resolution_layers(input_size,  min_feature_res):
        counter=1
        while input_size[0] and input_size[1] >= min_feature_res*2:
            counter+=1
            input_size = [x/2 for x in input_size]
        return counter

    @staticmethod
    def worker_init_fn(worker_id):
        imgaug.seed(np.random.get_state()[1][0] + worker_id)




