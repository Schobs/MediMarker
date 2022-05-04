
import os
import torch
import numpy as np
from time import time
# import multiprocessing as mp
from utils.im_utils.heatmap_manipulation import get_coords
from torch.cuda.amp import GradScaler, autocast

from torchvision.transforms import Resize,InterpolationMode
from datasets.dataset import DatasetBase
from torch.utils.data import DataLoader

from abc import ABC, abstractmethod
import imgaug

class NetworkTrainer(ABC):
    """ Super class for trainers. I extend this for trainers for U-Net and PHD-Net. They share some functions.y
    """
    @abstractmethod
    def __init__(self):
        #Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.network = None

        #early stopping
        self.early_stop_patience = 150

        #set this in init
        self.trainer_config = None

        #Trainer variables
        self.perform_validation = None
        self.continue_checkpoint =None
        self.logger = None
        self.verbose_logging = None

        self.profiler = None
        self.auto_mixed_precision = None
        self.amp_grad_scaler = None 


        self.was_initialized = False
        self.fold= None
        self.output_folder = None

      
        #Dataloader info
        self.data_loader_batch_size = 12
        self.num_batches_per_epoch = 150

        #Training params
        self.max_num_epochs =  1000
        self.initial_lr = 1e-2

        #Sigma for Gaussian heatmaps
        self.regress_sigma = False
        self.sigmas = None

        
        #get validaiton params
        self.use_full_res_coords = True
        self.resize_first = True

       
        #get model config parameters
        self.num_out_heatmaps = None
        self.input_size = None
        self.orginal_im_size =None

       
        self.num_input_channels = 1


        self.weight_inititialiser = None
        self.optimizer= None

        self.loss = None
     
        self.train_dataloader = self.valid_dataloader = None

        self.gen_hms_in_mainthread = False
      

        ################# Settings for saving checkpoints ##################################
        self.save_every = 25
        self.save_latest_only = False # if false it will not store/overwrite _latest but separate files each
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
        self.best_valid_coord_error = 999999999999999999999999999
        self.best_valid_loss_epoch = 0
        self.epochs_wo_val_improv = 0

    def initialize(self, training_bool=True):
        '''
        Initialize profiler, comet logger, training/val dataloaders, network, optimizer, loss, automixed precision
        and maybe load a checkpoint. 
        
        '''
        # torch.backends.cudnn.benchmark = True

        if self.profiler:
            print("Initialized profiler")
            self.profiler.start()
        
        if self.logger:
            self.logger.log_parameters(self.model_config)
            
        if training_bool:
            self.set_training_dataloaders()

        self.initialize_network()
        self.initialize_optimizer_and_scheduler()
        self.initialize_loss_function()
        self._maybe_init_amp()
        self.was_initialized = True

        self.maybe_load_checkpoint() 

    @abstractmethod
    def initialize_network(self):
        '''
        Initialize the network here!
        
        '''
      
    @abstractmethod
    def initialize_optimizer_and_scheduler(self):

        '''
        Initialize the optimizer and LR scheduler here!
        
        '''

    @abstractmethod
    def initialize_loss_function(self):
        '''
        Initialize the loss function here!
        
        '''


   

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
        print("initialized auto mixed precision.")

    @abstractmethod
    def get_coords_from_model_output(self, model_output):

        """
        Function to take model output and return coordinates.
        """


    def train(self):
        if not self.was_initialized:
            self.initialize(True)

        step = 0
        while self.epoch < self.max_num_epochs:
            

            self.epoch_start_time = time()
            train_losses_epoch = []

            self.network.train()

            generator = iter(self.train_dataloader)


            # Train for X number of batches per epoch e.g. 250
            for iter_b in range(self.num_batches_per_epoch):

                l, generator = self.run_iteration(generator, self.train_dataloader, backprop=True)
                train_losses_epoch.append(l)
                if self.logger:
                    self.logger.log_metric("training loss iteration", l, step)
                step += 1
            del generator
            self.all_tr_losses.append(np.mean(train_losses_epoch))

            
            with torch.no_grad():

                self.network.eval()
               
                val_coord_errors = []
                val_losses_epoch = []
                generator = iter(self.valid_dataloader)
                for iter_b in range(int(len(self.valid_dataloader.dataset)/self.data_loader_batch_size)):

                    l, generator = self.run_iteration(generator, self.valid_dataloader, backprop=False, get_coord_error=True, coord_error_list=val_coord_errors)
                    val_losses_epoch.append(l)
                 
                self.all_valid_losses.append(np.mean(val_losses_epoch)) 
                self.all_valid_coords.append(np.mean(val_coord_errors))

            self.epoch_end_time = time()

            #Print Information
            print("Epoch: ", self.epoch, " - train loss: ", np.mean(train_losses_epoch), " - val loss: ", self.all_valid_losses[-1], " - val coord error: ", self.all_valid_coords[-1], " -time: ",(self.epoch_end_time - self.epoch_start_time) )
            if self.regress_sigma:
                print("Sigmas: ", self.sigmas)

            continue_training = self.on_epoch_end()

            if not continue_training:
                if self.profiler:
                    self.profiler.stop()
                break
     
            self.epoch +=1

        #Save the final weights
        if self.logger:
            print("Logging weights as histogram...")
            weights = []
            for name in self.network.named_parameters():
                if 'weight' in name[0]:
                    weights.extend(name[1].detach().cpu().numpy().tolist())
            self.logger.log_histogram_3d(weights, step=self.epoch)



    @abstractmethod
    def predict_heatmaps_and_coordinates(self, data_dict,  return_all_layers = False, resize_to_og=False,):
        """ For inference. Predict heatmap and coordinates directly.

        Args:
            data_dict (_type_): _description_
            return_all_layers (bool, optional): _description_. Defaults to False.
            resize_to_og (bool, optional): _description_. Defaults to False.
        """


    def run_iteration(self, generator, dataloader, backprop, get_coord_error=False, coord_error_list=None):
        so = time()
        try:
            data_dict = next(generator)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            print("restarting generator")
            generator = iter(dataloader)
            data_dict = next(generator)


        data =(data_dict['image']).to( self.device )

        #This happens when we regress sigma with >0 workers due to multithreading issues.
        if self.gen_hms_in_mainthread:
            batch_hms = []
            # print("generating heatmaps in the main thread instead.")
            np_sigmas = [x.cpu().detach().numpy() for x in self.sigmas]
            # print("targ coordinates shape ", data_dict["target_coords"].shape)

            #b_ is 12,5, x, x but should be 5 long list of tensors: torch.Size([12, 19, X, X])

            b_= [dataloader.dataset.generate_labels(x, np_sigmas) for x in data_dict["target_coords"].detach().numpy()]

            for x in b_:
                if batch_hms == []:
                    batch_hms = [[y] for y in x]
                else:
                    for hm_idx, hm in enumerate(x):
                        # print(hm_idx, hm.shape   )
                        batch_hms[hm_idx].append(hm)

            batch_hms = [torch.stack(x) for x in batch_hms]

            
            data_dict['label'] = batch_hms   
          

        target = [x.to(self.device) for x in data_dict['label']]

        self.optimizer.zero_grad()
        from_which_level_supervision = self.num_res_supervision 


        if self.auto_mixed_precision:
            with autocast():
                if self.deep_supervision:
                    output = self.network(data)[-from_which_level_supervision:]                
                else:
                    output = self.network(data)
                del data
                l = self.loss(output, target, self.sigmas)
            if backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.learnable_params, 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
                if self.regress_sigma:
                    self.update_dataloader_sigmas(self.sigmas)

        else:
            if self.deep_supervision:
                output = self.network(data)[-from_which_level_supervision:]
            else:
                output = self.network(data)


            del data
            l = self.loss(output, target, self.sigmas)
        
            if backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.learnable_params, 12)
                self.optimizer.step() 
                if self.regress_sigma:
                    self.update_dataloader_sigmas(self.sigmas)


        if get_coord_error:
            with torch.no_grad():

                pred_coords = self.get_coords_from_model_output(output)
               
                if self.use_full_res_coords:
                    target_coords =data_dict['full_res_coords'].to( self.device )
                else:
                    target_coords =data_dict['target_coords'].to( self.device )


                if self.use_full_res_coords and not self.resize_first :
                    downscale_factor = [self.model_config.DATASET.ORIGINAL_IMAGE_SIZE[0]/self.model_config.SAMPLER.INPUT_SIZE[0], self.model_config.DATASET.ORIGINAL_IMAGE_SIZE[1]/self.model_config.SAMPLER.INPUT_SIZE[1]]
                    pred_coords = torch.rint(pred_coords * downscale_factor)

                coord_error = torch.linalg.norm((pred_coords- target_coords), axis=2)
                coord_error_list.append(np.mean(coord_error.detach().cpu().numpy()))

       

        if self.profiler:
            self.profiler.step()

        del output
        del target
        return l.detach().cpu().numpy(), generator


    def on_epoch_end(self):
        """
         Always run to 1000 epochs
        :return:
        """

        new_best_valid = False
        new_best_coord_valid = False

        continue_training = self.epoch < self.max_num_epochs

        if self.all_valid_losses[-1] < self.best_valid_loss:
            self.best_valid_loss = self.all_valid_losses[-1]
            self.best_valid_loss_epoch = self.epoch
            new_best_valid = True

        if self.all_valid_coords[-1] < self.best_valid_coord_error:
            self.best_valid_coord_error = self.all_valid_coords[-1]
            self.best_valid_coords_epoch = self.epoch
            new_best_coord_valid = True
            self.epochs_wo_val_improv = 0
        else:
            self.epochs_wo_val_improv += 1


            
        if self.epochs_wo_val_improv == self.early_stop_patience:
            continue_training = False
            print("EARLY STOPPING. Validation Coord Error did not reduce for %s epochs. " % self.early_stop_patience)

        self.maybe_save_checkpoint(new_best_valid, new_best_coord_valid)

        self.maybe_update_lr(epoch=self.epoch)

        # if self.regress_sigma:
        #     self.update_dataloader_sigmas(self.sigmas)

        if self.logger:
            self.logger.log_metric("training loss epoch", self.all_tr_losses[-1], self.epoch)
            self.logger.log_metric("validation loss", self.all_valid_losses[-1], self.epoch)
            self.logger.log_metric("validation coord error", self.all_valid_coords[-1], self.epoch)
            self.logger.log_metric("epoch time", (self.epoch_end_time - self.epoch_start_time), self.epoch)
            self.logger.log_metric("Learning rate", self.optimizer.param_groups[0]['lr'] , self.epoch)
            self.logger.log_metric("first_sigma", self.sigmas[0].cpu().detach().numpy() , self.epoch)

        
       
        return continue_training


    def maybe_save_checkpoint(self, new_best_valid_bool, new_best_valid_coord_bool):
        """
        Saves a checkpoint every save_ever epochs.
        :return:
        """

        fold_str = str(self.model_config.TRAINER.FOLD)
        if (self.save_intermediate_checkpoints and (self.epoch % self.save_every == (self.save_every - 1))) or self.epoch== self.max_num_epochs-1:
            print("saving scheduled checkpoint file...")
            if not self.save_latest_only:
                self.save_checkpoint(os.path.join(self.output_folder, "model_ep_"+ str(self.epoch) + "_fold" + fold_str+ ".model" % ()))
            
                if self.epoch >=150:
                    self.save_every = 50
                if self.epoch >=250:
                    self.save_every = 100
            self.save_checkpoint(os.path.join(self.output_folder, "model_latest_fold"+ (fold_str) +".model"))
            print("done")
        if new_best_valid_bool:
            print("saving scheduled checkpoint file as it's new best on validation set...")
            self.save_checkpoint(os.path.join(self.output_folder, "model_best_valid_loss_fold" + fold_str +".model"))

            print("done")

        if new_best_valid_coord_bool:
            print("saving scheduled checkpoint file as it's new best on validation set for coord error...")
            self.save_checkpoint(os.path.join(self.output_folder, "model_best_valid_coord_error_fold" + fold_str +".model"))

            print("done")

           






    def save_checkpoint(self, path):
        state = {
            'epoch': self.epoch + 1,
            'state_dict': self.network.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'best_valid_loss': self.best_valid_loss,
            'best_valid_coord_error': self.best_valid_coord_error,
            'best_valid_loss_epoch': self.best_valid_loss_epoch,
            "best_valid_coords_epoch": self.best_valid_coords_epoch,
            "epochs_wo_improvement": self.epochs_wo_val_improv,
            "sigmas": self.sigmas
        }

        if self.amp_grad_scaler is not None:
            state['amp_grad_scaler'] = self.amp_grad_scaler.state_dict()

        torch.save(state, path)


    def maybe_load_checkpoint(self):
        if self.continue_checkpoint:
            self.load_checkpoint(self.continue_checkpoint, True)
    
    def update_dataloader_sigmas(self, new_sigmas):
        np_sigmas = [x.cpu().detach().numpy() for x in new_sigmas]
        self.train_dataloader.dataset.sigmas = (np_sigmas)
        self.valid_dataloader.dataset.sigmas = (np_sigmas)



    def load_checkpoint(self, model_path, training_bool):
        if not self.was_initialized:
            self.initialize(training_bool)

        checkpoint_info = torch.load(model_path, map_location=self.device)
        self.epoch = checkpoint_info['epoch']
        self.network.load_state_dict(checkpoint_info["state_dict"])
        self.optimizer.load_state_dict(checkpoint_info["optimizer"])
        self.best_valid_loss = checkpoint_info['best_valid_loss']
        self.best_valid_loss_epoch = checkpoint_info['best_valid_loss_epoch']
        self.best_valid_coord_error = checkpoint_info['best_valid_coord_error']
        self.best_valid_coords_epoch = checkpoint_info["best_valid_coords_epoch"]
        self.epochs_wo_val_improv = checkpoint_info["epochs_wo_improvement"]
        self.sigmas= checkpoint_info["sigmas"]


        if self.auto_mixed_precision:
            self._maybe_init_amp()

            if 'amp_grad_scaler' in checkpoint_info.keys():
                self.amp_grad_scaler.load_state_dict(checkpoint_info['amp_grad_scaler'])

        print("Loaded checkpoint %s. Epoch: %s, " % (model_path, self.epoch ))

    # @abstractmethod
    def set_training_dataloaders(self):
        """
        set train_dataset, valid_dataset and train_dataloader and valid_dataloader here.
        """

        np_sigmas = [x.cpu().detach().numpy() for x in self.sigmas]
    
        train_dataset = DatasetBase(
            annotation_path =self.model_config.DATASET.SRC_TARGETS,
            landmarks = self.model_config.DATASET.LANDMARKS,
            LabelGenerator = self.label_generator,
            split = "training",
            sample_patches = self.model_config.SAMPLER.SAMPLE_PATCH,
            sample_patch_size = self.model_config.SAMPLER.SAMPLE_PATCH_SIZE,
            sample_patch_bias = self.model_config.SAMPLER.SAMPLER_BIAS,
            root_path = self.model_config.DATASET.ROOT,
            sigmas =  np_sigmas,
            generate_hms_here = not self.model_config.INFERRED_ARGS.GEN_HM_IN_MAINTHREAD, 
            cv = self.model_config.TRAINER.FOLD,
            cache_data = self.model_config.TRAINER.CACHE_DATA,
            num_res_supervisions = self.num_res_supervision,
            debug=self.model_config.SAMPLER.DEBUG ,
            original_image_size= self.model_config.DATASET.ORIGINAL_IMAGE_SIZE,
            input_size =  self.model_config.SAMPLER.INPUT_SIZE,
            hm_lambda_scale = self.model_config.MODEL.HM_LAMBDA_SCALE,
            data_augmentation_strategy = self.model_config.SAMPLER.DATA_AUG,
            data_augmentation_package = self.model_config.SAMPLER.DATA_AUG_PACKAGE,

 
        )


        if self.perform_validation:
            val_split= "validation"
            
            valid_dataset = DatasetBase(
                annotation_path =self.model_config.DATASET.SRC_TARGETS,
                landmarks = self.model_config.DATASET.LANDMARKS,
                LabelGenerator = self.label_generator,
                split = val_split,
                sample_patches = self.model_config.SAMPLER.SAMPLE_PATCH,
                sample_patch_size = self.model_config.SAMPLER.SAMPLE_PATCH_SIZE,
                sample_patch_bias = self.model_config.SAMPLER.SAMPLER_BIAS,
                root_path = self.model_config.DATASET.ROOT,
                sigmas =  np_sigmas,
                generate_hms_here = not self.model_config.INFERRED_ARGS.GEN_HM_IN_MAINTHREAD, 
                cv = self.model_config.TRAINER.FOLD,
                cache_data = self.model_config.TRAINER.CACHE_DATA,
                num_res_supervisions = self.num_res_supervision,
                debug=self.model_config.SAMPLER.DEBUG,
                data_augmentation_strategy =None,
                original_image_size= self.model_config.DATASET.ORIGINAL_IMAGE_SIZE,
                input_size =  self.model_config.SAMPLER.INPUT_SIZE,
                hm_lambda_scale = self.model_config.MODEL.HM_LAMBDA_SCALE,

            )
        else:
            val_split = "training"
            valid_dataset = train_dataset
            print("WARNING: NOT performing validation. Instead performing \"validation\" on training set for coord error metrics.")



        #If debug use only main thread to load data bc we only want to show a single plot on screen.
        #If num_workers=0 we are only using the main thread, so persist_workers = False.
        if self.model_config.SAMPLER.DEBUG or self.model_config.SAMPLER.NUM_WORKERS == 0:
            persist_workers = False
            num_workers_cfg=0
        else:
            persist_workers = True
            num_workers_cfg= self.model_config.SAMPLER.NUM_WORKERS    
       

        self.train_dataloader = DataLoader(train_dataset, batch_size=self.model_config.SOLVER.DATA_LOADER_BATCH_SIZE, shuffle=True, num_workers=num_workers_cfg, persistent_workers=persist_workers, worker_init_fn=NetworkTrainer.worker_init_fn, pin_memory=True )
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=self.model_config.SOLVER.DATA_LOADER_BATCH_SIZE, shuffle=False, num_workers=num_workers_cfg, persistent_workers=persist_workers, worker_init_fn=NetworkTrainer.worker_init_fn, pin_memory=True )
    
    @staticmethod
    def worker_init_fn(worker_id):
        imgaug.seed(np.random.get_state()[1][0] + worker_id)

