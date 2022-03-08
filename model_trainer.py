
from torch import nn
import os
from initialization import InitWeights_KaimingUniform
from losses import HeatmapLoss, IntermidiateOutputLoss, AdaptiveWingLoss
from model import UNet
import torch
import numpy as np
from time import time
from dataset import ASPIRELandmarks
from torch.utils.data import DataLoader
from utils.heatmap_manipulation import get_coords
from torch.cuda.amp import GradScaler, autocast

class UnetTrainer():
    """ Class for the u-net trainer stuff.
    """

    def __init__(self, fold, model_config= None, output_folder=None, logger=None, profiler=None):
        
        #config variable
        self.model_config = model_config

        self.continue_checkpoint = model_config.MODEL.CHECKPOINT
        self.logger = logger
        self.profiler = profiler
        self.verbose_logging = model_config.OUTPUT.VERBOSE
        self.auto_mixed_precision = model_config.SOLVER.AUTO_MIXED_PRECISION
        self.amp_grad_scaler = None #initialise later 
        self.was_initialized = False
        self.fold= fold
        self.output_folder = output_folder
        self.pin_memory = True

        #Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #Dataloader info
        self.data_loader_batch_size = model_config.SOLVER.DATA_LOADER_BATCH_SIZE
        self.num_val_batches_per_epoch = 50
        self.num_batches_per_epoch = model_config.SOLVER.MINI_BATCH_SIZE

        #Training params
        self.max_num_epochs =  model_config.SOLVER.MAX_EPOCHS
        self.initial_lr = 1e-2


      

        #get model config parameters
        self.num_out_heatmaps = len(model_config.DATASET.LANDMARKS)
        self.base_num_features = model_config.MODEL.INIT_FEATURES
        self.max_features = model_config.MODEL.MAX_FEATURES
        self.input_size = model_config.DATASET.INPUT_SIZE
        # self.num_resolution_layers = 8
        self.num_resolution_layers = UnetTrainer.get_resolution_layers(self.input_size)

        print("num res layers = ", self.num_resolution_layers)


        self.num_input_channels = 1
        self.num_downsampling = 7
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
            self.individual_loss = HeatmapLoss()
        elif loss_str =="awl":
            self.individual_loss = AdaptiveWingLoss()
        else:
            raise ValueError("the loss function %s is not implemented. Try mse or awl" % (loss_str))



      

        ################# Settings for saving checkpoints ##################################
        self.save_every = 50
        self.save_latest_only = True  # if false it will not store/overwrite _latest but separate files each
        # time an intermediate checkpoint is created
        self.save_intermediate_checkpoints = True  # whether or not to save checkpoint_latest
        self.save_best_checkpoint = True  # whether or not to save the best checkpoint according to self.best_val_eval_criterion_MA
        self.save_final_checkpoint = True  # whether or not to save the final checkpoint

         #variables to save to.
        self.all_tr_losses = []
        self.all_valid_losses = []
        self.all_valid_coords = []

        #Initialize
        self.epoch = 0
        self.best_valid_loss = 999999999999999999999999999
        self.best_valid_epoch = 0


    def initialize(self, training_bool=True):

        # torch.backends.cudnn.benchmark = True

        if self.profiler:
            self.profiler.start()
            
        if training_bool:
            self.set_training_dataloaders()

        self.initialize_network()
        self.initialize_optimizer_and_scheduler()
        self.initialize_loss_function()
        self._maybe_init_amp()
        self.was_initialized = True

        self.maybe_load_checkpoint() 


    def initialize_network(self):

        # Let's make the network
        self.network = UNet(input_channels=self.num_input_channels, base_num_features=self.base_num_features, num_out_heatmaps=self.num_out_heatmaps,
            num_resolution_levels= self.num_resolution_layers, conv_operation=self.conv_operation, normalization_operation=self.normalization_operation,
            normalization_operation_config=self.norm_op_kwargs, activation_function= self.activation_function, activation_func_config= self.activation_kwargs,
            weight_initialization=self.weight_inititialiser, strided_convolution_kernels = self.pool_op_kernel_size, convolution_kernels= self.conv_op_kernel_size,
            convolution_config=self.conv_kwargs, upsample_operation=self.upsample_operation, max_features=self.max_features, deep_supervision=self.deep_supervision
        
         )


        if torch.cuda.is_available():
            self.network.cuda()

    
    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = self.optimizer(self.network.parameters(), **self.optimizer_kwargs)



    def initialize_loss_function(self):

        if self.deep_supervision:
            #first get weights for the layers. We don't care about the first two decoding levels
            weights = np.array([1 / (2 ** i) for i in range(self.num_resolution_layers-1)])[::-1] #-1 because we don't use bottleneck layer. reverse bc the high res ones are important
            # print("weights before: ", weights)
            loss_weights = np.array([0 if idx>=(len(weights)-2) else x for (idx,x) in enumerate(weights)]) #ignore lowest two resolutions, assign 0 weight to them
            # print("weights after mask: ", loss_weights)
            loss_weights = (loss_weights / loss_weights.sum()) #Normalise to add to 1
            # print("weights after normalize: ", loss_weights)

            self.loss = IntermidiateOutputLoss(self.individual_loss, loss_weights)
        else:
            self.loss = IntermidiateOutputLoss(self.individual_loss, [1])

    def maybe_update_lr(self, epoch=None, exponent=0.9):
        """
        if epoch is not None we overwrite epoch. Else we use epoch = self.epoch + 1

        (maybe_update_lr is called in on_epoch_end which is called before epoch is incremented.
        Therefore we need to do +1 here)

        """
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch
        poly_lr_update = self.initial_lr * (1 - ep / self.max_num_epochs)**exponent

        self.optimizer.param_groups[0]['lr'] =poly_lr_update

    def _maybe_init_amp(self):
        if self.auto_mixed_precision and self.amp_grad_scaler is None:
            self.amp_grad_scaler = GradScaler()
   

    def train(self):
        while self.epoch < self.max_num_epochs:
            self.epoch_start_time = time()
            train_losses_epoch = []

            self.network.train()
            generator = iter(self.train_dataloader)
            #Train for X number of batches per epoch e.g. 250
            for _ in range(self.num_batches_per_epoch):

                e = time()
                l = self.run_iteration(generator, self.train_dataloader, backprop=True)

                # print("1 iter time: ", time()-e)
                # print(".", l, end= "")

                train_losses_epoch.append(l)

            self.all_tr_losses.append(np.mean(train_losses_epoch))

            #Validate
            with torch.no_grad():
                self.network.eval()
                # val_losses = []
                val_coord_errors = []
                generator = iter(self.valid_dataloader)

                # for b in range(self.num_val_batches_per_epoch):
                l = self.run_iteration(generator, self.valid_dataloader, False, True, val_coord_errors)
                # val_losses.append(l)
                # print(l, len(self.valid_dataloader.dataset), l/len(generator))

                self.all_valid_losses.append(l/len(self.valid_dataloader.dataset)) #go through all valid at once here
                self.all_valid_coords.append(np.mean(val_coord_errors))


            self.epoch_end_time = time()
            continue_training = self.on_epoch_end()

            if not continue_training:
                # allows for early stopping
                self.logger.flush()
                self.logger.close()

                if self.profiler:
                    self.profiler.stop()

                break
            

            print("Epoch: ", self.epoch, " - train loss: ", np.mean(train_losses_epoch), " - val loss: ", self.all_valid_losses[-1], " - val coord error: ", self.all_valid_coords[-1], " -time: ",(self.epoch_end_time - self.epoch_start_time) )

            self.epoch +=1

    def predict_heatmaps_and_coordinates(self, data_dict, return_all_layers = False):
        data =(data_dict['image']).to( self.device )
        target = [x.to(self.device) for x in data_dict['label']]
        from_which_level_supervision = self.num_downsampling - self.num_res_supervision 

        if self.deep_supervision:
            output = self.network(data)[from_which_level_supervision:]
        else:
            output = self.network(data)

        
        l = self.loss(output, target)

        predicted_coords = get_coords(output[-1])

        heatmaps_return = output
        if not return_all_layers:
            heatmaps_return = output[-1] #only want final layer


        return heatmaps_return, predicted_coords, l.detach().cpu().numpy()


    def run_iteration(self, generator, dataloader, backprop, get_coord_error=False, coord_error_list=None):
        so = time()
        try:
            # Samples the batch
            data_dict = next(generator)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            generator = iter(dataloader)
            data_dict = next(generator)

        data =(data_dict['image']).to( self.device )
        target = [x.to(self.device) for x in data_dict['label']]
        # print("get data time ", time()-so)

        #write to tensorboard if verbose if first epoch
        if self.verbose_logging and self.epoch==0:
            self.logger.add_graph(self.network, data)
      
        self.optimizer.zero_grad()

        from_which_level_supervision = self.num_downsampling - self.num_res_supervision 

        # print("from which level supervision:@ ", from_which_level_supervision)
        s= time()

        if self.auto_mixed_precision:
            with autocast():
                if self.deep_supervision:
                    output = self.network(data)[from_which_level_supervision:]
                else:
                    #need to turn them from list into single tensor for loss function
                    output = self.network(data)
                    # target = target[0] 


                # print("lens of them", len(output), len(target))
                del data
                l = self.loss(output, target)

            if backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            if self.deep_supervision:
                output = self.network(data)[from_which_level_supervision:]
            else:
                #need to turn them from list into single tensor for loss function
                output = self.network(data)
                # target = target[0] 


            del data
            l = self.loss(output, target)
        
            s=time()
            if backprop:
                sa = time()
                l.backward()
                # print("loss back time ", time()-sa)
                sa = time()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12) #this taking olong. 0.085 secs , now 0.15 secs???
                # print("grad clip time ", time()-sa)
                sa = time()
                self.optimizer.step() #optimizer taking long: 0.21 secs , now 0.338 secs??
                # print("optimizer time ", time()-sa)

        # print("back prop time ", time()-s)

        # print("itertime ", time()-so)
        if get_coord_error:
            with torch.no_grad():

                predicted_coords = get_coords(output[-1])
                target_coords = torch.as_tensor(data_dict['target_coords']).to(self.device)

                # print("pred %s and targ %s : " % (predicted_coords, target_coords))
                coord_error = torch.linalg.norm((predicted_coords- target_coords), axis=2).detach().cpu().numpy()
                # print("coord errors: ", coord_error)
                coord_error_list.append(np.mean(coord_error))

            # raise NotImplementedError("have not implented coord error yet.")

        if self.profiler:
            self.profiler.step()

        del target
        return l.detach().cpu().numpy()


    def on_epoch_end(self):
        """
         Always run to 1000 epochs
        :return:
        """

        new_best_valid = False

        continue_training = self.epoch < self.max_num_epochs

        if self.all_valid_losses[-1] < self.best_valid_loss:
            self.best_valid_loss = self.all_valid_losses[-1]
            self.best_valid_epoch = self.epoch
            new_best_valid = True
        
        self.maybe_save_checkpoint(new_best_valid)

        self.maybe_update_lr(epoch=self.epoch)

        self.logger.add_scalar("training loss", self.all_tr_losses[-1], self.epoch)
        self.logger.add_scalar("validation loss", self.all_valid_losses[-1], self.epoch)
        self.logger.add_scalar("validation coord error", self.all_valid_coords[-1], self.epoch)
        self.logger.add_scalar("epoch time", (self.epoch_end_time - self.epoch_start_time), self.epoch)
        self.logger.add_scalar("Learning rate", self.optimizer.param_groups[0]['lr'] , self.epoch)

        
       
        return continue_training


    def maybe_save_checkpoint(self, new_best_valid_bool):
        """
        Saves a checkpoint every save_ever epochs.
        :return:
        """
        if self.save_intermediate_checkpoints and (self.epoch % self.save_every == (self.save_every - 1)):
            print("saving scheduled checkpoint file...")
            if not self.save_latest_only:
                self.save_checkpoint(os.path.join(self.output_folder, "model_ep_%03.0d.model" % (self.epoch)))
                
            self.save_checkpoint(os.path.join(self.output_folder, "model_latest.model"))
            print("done")
        if new_best_valid_bool:
            print("saving scheduled checkpoint file as it's new best on validation set...")
            self.save_checkpoint(os.path.join(self.output_folder, "model_best_valid.model"))
            print("done")





    def save_checkpoint(self, path):
        state = {
            'epoch': self.epoch + 1,
            'state_dict': self.network.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'best_valid_loss': self.best_valid_loss,
            'best_valid_epoch': self.best_valid_epoch

        }
        if self.amp_grad_scaler is not None:
            state['amp_grad_scaler'] = self.amp_grad_scaler.state_dict()

        torch.save(state, path)


    def maybe_load_checkpoint(self):
        if self.continue_checkpoint:
            self.load_checkpoint(self.continue_checkpoint, True)

    def load_checkpoint(self, model_path, training_bool):
        if not self.was_initialized:
            self.initialize(training_bool)

        checkpoint_info = torch.load(model_path, map_location=self.device)
        self.epoch = checkpoint_info['epoch']
        self.network.load_state_dict(checkpoint_info["state_dict"])
        self.optimizer.load_state_dict(checkpoint_info["optimizer"])
        self.best_valid_loss = checkpoint_info['best_valid_loss']
        self.best_valid_epoch = checkpoint_info['best_valid_epoch']

        if self.auto_mixed_precision:
            self._maybe_init_amp()

            if 'amp_grad_scaler' in checkpoint_info.keys():
                self.amp_grad_scaler.load_state_dict(checkpoint_info['amp_grad_scaler'])



    def set_training_dataloaders(self):
    
        train_dataset = ASPIRELandmarks(
            annotation_path =self.model_config.DATASET.SRC_TARGETS,
            landmarks = self.model_config.DATASET.LANDMARKS,
            split = "training",
            root_path = self.model_config.DATASET.ROOT,
            sigma = self.model_config.MODEL.GAUSS_SIGMA,
            cv = 1,
            cache_data = True,
            normalize=True,
            num_res_supervisions = self.num_res_supervision,
            debug=self.model_config.DATASET.DEBUG ,
            data_augmentation =self.model_config.DATASET.DATA_AUG,
            original_image_size= self.model_config.DATASET.ORIGINAL_IMAGE_SIZE,
            input_size =  self.model_config.DATASET.INPUT_SIZE
 
        )

        valid_dataset = ASPIRELandmarks(
            annotation_path =self.model_config.DATASET.SRC_TARGETS,
            landmarks = self.model_config.DATASET.LANDMARKS,
            split = "validation",
            root_path = self.model_config.DATASET.ROOT,
            sigma = self.model_config.MODEL.GAUSS_SIGMA,
            cv = 1,
            cache_data = True,
            normalize=True,
            num_res_supervisions = self.num_res_supervision,
            debug=self.model_config.DATASET.DEBUG,
            data_augmentation =self.model_config.DATASET.DATA_AUG,
            original_image_size= self.model_config.DATASET.ORIGINAL_IMAGE_SIZE,
            input_size =  self.model_config.DATASET.INPUT_SIZE
 



        )

        self.train_dataloader = DataLoader(train_dataset, batch_size=self.model_config.SOLVER.DATA_LOADER_BATCH_SIZE, shuffle=True)
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=False)

    @staticmethod
    def get_resolution_layers(input_size):
        counter=1
        while input_size[0] and input_size[1] >= 8:
            counter+=1
            input_size = [x/2 for x in input_size]
        return counter