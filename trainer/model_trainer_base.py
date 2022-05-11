
import os
import torch
import numpy as np
from time import time
# import multiprocessing as mp
from utils.im_utils.heatmap_manipulation import get_coords
from torch.cuda.amp import GradScaler, autocast
from losses import HeatmapLoss, IntermediateOutputLoss, AdaptiveWingLoss, SigmaLoss

from torchvision.transforms import Resize,InterpolationMode
from datasets.dataset import DatasetBase
from torch.utils.data import DataLoader

from abc import ABC, abstractmethod
import imgaug
import copy

class NetworkTrainer(ABC):
    """ Super class for trainers. I extend this for trainers for U-Net and PHD-Net. They share some functions.y
    """
    @abstractmethod
    def __init__(self, trainer_config,  is_train= True, output_folder=None, logger=None, profiler=None):
        #Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #This is the trainer config dict
        self.trainer_config = trainer_config
        self.is_train = is_train

        #Dataloader info
        self.data_loader_batch_size = self.trainer_config.SOLVER.DATA_LOADER_BATCH_SIZE
        self.num_batches_per_epoch = self.trainer_config.SOLVER.MINI_BATCH_SIZE
        self.gen_hms_in_mainthread = self.trainer_config.INFERRED_ARGS.GEN_HM_IN_MAINTHREAD
        self.sampler_mode = self.trainer_config.SAMPLER.SAMPLE_MODE
        self.landmarks = self.trainer_config.DATASET.LANDMARKS
       

        #Set up logger & profiler
        self.profiler = profiler
        self.logger = logger
        self.verbose_logging = self.trainer_config.OUTPUT.VERBOSE


        #Set up directories
        self.output_folder = output_folder

        #Trainer variables
        self.perform_validation = self.trainer_config.TRAINER.PERFORM_VALIDATION
        self.fold= self.trainer_config.TRAINER.FOLD
        self.continue_checkpoint = self.trainer_config.MODEL.CHECKPOINT
        
        self.auto_mixed_precision = self.trainer_config.SOLVER.AUTO_MIXED_PRECISION

        #Training params
        self.max_num_epochs =   self.trainer_config.SOLVER.MAX_EPOCHS
        #Regressing sigma parameters for heatmaps
        self.regress_sigma = self.trainer_config.SOLVER.REGRESS_SIGMA
        self.sigmas = [torch.tensor(x, dtype=float, device=self.device, requires_grad=True) for x in np.repeat(self.trainer_config.MODEL.GAUSS_SIGMA, len(self.landmarks))]

        #Validation parameters
        self.use_full_res_coords = self.trainer_config.INFERRED_ARGS.USE_FULL_RES_COORDS
        self.resize_first = self.trainer_config.INFERRED_ARGS.RESIZE_FIRST 
        

        #Checkpointing params
        self.save_every = 25
        self.save_latest_only = self.trainer_config.TRAINER.SAVE_LATEST_ONLY # if false it will not store/overwrite _latest but separate files each
        self.save_intermediate_checkpoints = True  # whether or not to save checkpoint_latest

        #Loss function

      
       
        #To be initialised in the super class (here)
        self.was_initialized = False
        self.amp_grad_scaler = None 
        self.train_dataloader = self.valid_dataloader = None
        self.logged_per_epoch = None


        #Set up in init of extended class (child)
        self.network = None
        self.label_generator = None
        self.optimizer= None
        self.loss = None
        self.num_res_supervision = None

        #Can be changed in extended class (child)
        self.early_stop_patience = 150
        self.initial_lr = 1e-2
     

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
            self.logger.log_parameters(self.trainer_config)
            
        if training_bool:
            self.set_training_dataloaders()

        self.initialize_network()
        self.initialize_optimizer_and_scheduler()
        self.initialize_loss_function()
        self._maybe_init_amp()
        self.was_initialized = True

        self.maybe_load_checkpoint()

        self.initialize_keys_for_logging_epoch()

    @abstractmethod
    def initialize_network(self):
        '''
        Initialize the network here!
        
        '''
    
    @abstractmethod
    def initialize_keys_for_logging_epoch(self):
        '''
        Initialize  keys to log in self.logged_per_epoch here, especially logging the keys from the loss_dict 
        for proper loss logging!.These are used in self.log_key_variables() every epoch.
        Extend this in child class for other keys to log :)
        '''
        self.logged_per_epoch =  {"valid_coord_error_mean": [], "epoch_time": []}
        if self.regress_sigma:
            self.logged_per_epoch["sigmas_mean"] =  []
            for sig in range(len(self.landmarks)):
                 self.logged_per_epoch["sigma_"+str(sig)] = []


        for key_ in self.loss.loss_seperated_keys:
            self.logged_per_epoch["training_" + key_] = []
            self.logged_per_epoch["validation_" + key_] = []



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
        Function to take model output and return coordinates & a Dict of any extra information to log (e.g. max of heatmap)
       """

    # @abstractmethod
    # def run_inference(self):
    #     """ Function to run inference.
    #     """


    def train(self):
        if not self.was_initialized:
            self.initialize(True)

        step = 0
        while self.epoch < self.max_num_epochs:
            

            self.epoch_start_time = time()

            self.network.train()

            generator = iter(self.train_dataloader)

            #We will log the training and validation info here. The keys we set describe all the info we are logging.
            per_epoch_logs =  copy.deepcopy(self.logged_per_epoch)


            # Train for X number of batches per epoch e.g. 250
            for iter_b in range(self.num_batches_per_epoch):
                l, generator = self.run_iteration(generator, self.train_dataloader, backprop=True, split="training", log_coords=False,  logged_vars=per_epoch_logs)
                if self.logger:
                    self.logger.log_metric("training loss iteration", l, step)
                step += 1
            del generator

            
            with torch.no_grad():
                self.network.eval()
                generator = iter(self.valid_dataloader)
                for iter_b in range(int(len(self.valid_dataloader.dataset)/self.data_loader_batch_size)):
                    l, generator = self.run_iteration(generator, self.valid_dataloader, backprop=False, split="validation", log_coords=True,  logged_vars=per_epoch_logs)
                 
            
            self.epoch_end_time = time()
           
            continue_training = self.on_epoch_end(per_epoch_logs)

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


    def run_iteration(self, generator, dataloader, backprop, split, log_coords, logged_vars=None):
        so = time()
        try:
            data_dict = next(generator)
        except StopIteration:
            # restart the generator if the previous generator is exhausted. If valid/test then all data has been used
            if split != "training":
                return 0, None
            else:
                generator = iter(dataloader)
                data_dict = next(generator)


        data =(data_dict['image']).to( self.device )


        #This happens when we regress sigma with >0 workers due to multithreading issues.
        if self.gen_hms_in_mainthread:
            data_dict['label'] = self.generate_heatmaps_batch(data_dict, dataloader)          

        target = [x.to(self.device) for x in data_dict['label']]

        self.optimizer.zero_grad()


        if self.auto_mixed_precision:
            with autocast():
                output = self.network(data)
                del data
                l, loss_dict = self.loss(output, target, self.sigmas)
            if backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.learnable_params, 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
                if self.regress_sigma:
                    self.update_dataloader_sigmas(self.sigmas)

        else:
            output = self.network(data)
            del data
            l, loss_dict = self.loss(output, target, self.sigmas)
        
            if backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.learnable_params, 12)
                self.optimizer.step() 
                if self.regress_sigma:
                    self.update_dataloader_sigmas(self.sigmas)


        #Log info from this iteration.
        if list(logged_vars.keys()) != []:
            with torch.no_grad():
                self.log_key_variables(output, loss_dict, data_dict, logged_vars, log_coords, split)

        if self.profiler:
            self.profiler.step()

        del output
        del target
        return l.detach().cpu().numpy(), generator

    def generate_heatmaps_batch(self, data_dict, dataloader):
        batch_hms = []
        np_sigmas = [x.cpu().detach().numpy() for x in self.sigmas]
        b_= [dataloader.dataset.generate_labels(x, np_sigmas) for x in data_dict["target_coords"].detach().numpy()]
        for x in b_:
            if batch_hms == []:
                batch_hms = [[y] for y in x]
            else:
                for hm_idx, hm in enumerate(x):
                    batch_hms[hm_idx].append(hm)
        batch_hms = [torch.stack(x) for x in batch_hms]

        return batch_hms

        
    def on_epoch_end(self, per_epoch_logs):
        """
         Always run to 1000 epochs
        :return:
        """

        new_best_valid = False
        new_best_coord_valid = False

        continue_training = self.epoch < self.max_num_epochs

        #######Logging some end of epoch info #############
        if "epoch_time" in list(per_epoch_logs.keys()):
            per_epoch_logs["epoch_time"] = self.epoch_end_time - self.epoch_start_time
        if "sigmas_mean" in list(per_epoch_logs.keys()):
            np_sigmas = [x.cpu().detach().numpy() for x in self.sigmas]
            per_epoch_logs["sigmas_mean"] = (np.mean(np_sigmas))

            for idx, sig in enumerate(np_sigmas):
                if "sigma_"+str(idx) in list(per_epoch_logs.keys()):
                    per_epoch_logs["sigma_"+str(idx)] = sig

        for key, value in per_epoch_logs.items():
            #get the mean of all the batches from the training/validations. 
            if isinstance(value, list):
                per_epoch_logs[key] = np.mean([x.detach().cpu().numpy() if torch.is_tensor(x) else x for x in value])
            if torch.is_tensor(value):
                per_epoch_logs[key] = value.detach().cpu().numpy()
            
            #Log them if using a logger, else they are lost!
            if self.logger:
                self.logger.log_metric(key, per_epoch_logs[key], self.epoch)

        print("the final per epoch logs :", per_epoch_logs )


        #Checks for it this epoch was best in validation loss or validation coord error!
        if per_epoch_logs["validation_all_loss_all"] < self.best_valid_loss:
            self.best_valid_loss = per_epoch_logs["validation_all_loss_all"]
            self.best_valid_loss_epoch = self.epoch
            new_best_valid = True

        if per_epoch_logs["valid_coord_error_mean"] < self.best_valid_coord_error:
            self.best_valid_coord_error = per_epoch_logs["valid_coord_error_mean"]
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

       
        return continue_training


    def maybe_save_checkpoint(self, new_best_valid_bool, new_best_valid_coord_bool):
        """
        Saves a checkpoint every save_ever epochs.
        :return:
        """

        fold_str = str(self.fold)
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

           
    def log_key_variables(self, output, loss_dict, data_dict, logged_vars, log_coords, split):
        """Logs base key variables. Should be extended by child class for non-generic logging variables.

        Args:
            output (_type_): _description_
            loss (_type_): _description_
            data_dict (_type_): _description_
            logged_vars (_type_): _description_
        """

        #1) Log training/validation losses based on split.

        for key, value in loss_dict.items():
            key_ = split+ "_" + key
            assert key_ in logged_vars         

            logged_vars[key_].append(value)
            
        #2) If log_coords, get coords from output. Then, check for the keys for what to log.
        if log_coords:      
            #Get coordinates from output heatmap, comparing to GT compared on arguments. Cases C:
            # C1) resize heatmap, use full res coords
            # C2) don't resize heatmap, use full res coords (already using full scale image)
            # C3) don't resize heatmap, use downscaled coords
            # C4) don't resize heatmap, but extract pred_coords and upscale them
            pred_coords, extra_info = self.get_coords_from_model_output(output)
            
            if self.use_full_res_coords:
                target_coords =data_dict['full_res_coords'].to( self.device ) #C1, C2, C4
            else:
                target_coords =data_dict['target_coords'].to( self.device ) #C3

            # C4
            if self.use_full_res_coords and not self.resize_first :
                upscale_factor = [self.trainer_config.DATASET.ORIGINAL_IMAGE_SIZE[0]/self.trainer_config.SAMPLER.INPUT_SIZE[0], self.trainer_config.DATASET.ORIGINAL_IMAGE_SIZE[1]/self.trainer_config.SAMPLER.INPUT_SIZE[1]]
                pred_coords = torch.rint(pred_coords * upscale_factor)
                
            coord_error = torch.linalg.norm((pred_coords- target_coords), axis=2)
            

            if split == "validation":
                logged_vars["valid_coord_error_mean" ].append(np.mean(coord_error.detach().cpu().numpy()))
            elif split == "training":
                assert "train_coord_error_mean" in list(logged_vars.keys())
                logged_vars["train_coord_error_mean" ].append(np.mean(coord_error.detach().cpu().numpy()))


            #Logging any other extra info (e.g. heatmap max values), so check if the keys are requested in our logging dict.
            # In these cases we need to save each sample seperately
            extra_logging_keys = ["coord_error_all", "predicted_coords", "target_coords", "uid"] + list(extra_info.keys())
            if any(x in extra_logging_keys for x in list(logged_vars.keys()) ):
                for idx, sample in enumerate(output):
                    print("idx, sample shape", idx, sample.shape)
                    print("coord error for 1 sample: ", coord_error[idx])
                    if "coord_error_all" in list(logged_vars.keys()): 
                        logged_vars["coord_error_all"].append((coord_error[idx].detach().cpu().numpy()))
                    if "predicted_coords" in list(logged_vars.keys()): 
                        logged_vars["predicted_coords"].append((pred_coords[idx].detach().cpu().numpy()))
                    if "target_coords" in list(logged_vars.keys()): 
                        logged_vars["target_coords"].append((target_coords[idx].detach().cpu().numpy()))
                    if "uid" in list(logged_vars.keys()): 
                        logged_vars["uid"].append((data_dict["uid"][idx].detach().cpu().numpy()))

                    # for extra_info, ensure key exists in user's logged_vars 
                    for key_ in list(extra_info.keys()):
                        if key_ in list(logged_vars.keys()):
                            logged_vars[key_].append((extra_info["key_"][idx].detach().cpu().numpy()))
 
            

    def inference_patchified(self, samples_patchified, stitching_infos):
        raise NotImplementedError()
        
        all_hms = []
        for idx, sample_patches in enumerate(samples_patchified):
            stich_info = stitching_infos[idx]
            patch_predictions = []

            for patch in sample_patches:
                output = self.network(patch)
                patch_predictions.append(output)


            final_hm = self.stitch_heatmap(patch_predictions, stich_info)
            all_hms.append(torch.tensor(final_hm))

        all_hms = torch.stack(all_hms)
        print("hm patchified output len: ", all_hms.shape)
        return all_hms
      



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

    def set_training_dataloaders(self):
        """
        set train_dataset, valid_dataset and train_dataloader and valid_dataloader here.
        """

        np_sigmas = [x.cpu().detach().numpy() for x in self.sigmas]
    
        train_dataset = DatasetBase(
            annotation_path =self.trainer_config.DATASET.SRC_TARGETS,
            landmarks = self.landmarks,
            LabelGenerator = self.label_generator,
            split = "training",
            sample_mode = self.sampler_mode,
            sample_patch_size = self.trainer_config.SAMPLER.PATCH.SAMPLE_PATCH_SIZE,
            sample_patch_bias = self.trainer_config.SAMPLER.PATCH.SAMPLER_BIAS,
            sample_patch_from_resolution = self.trainer_config.SAMPLER.PATCH.RESOLUTION_TO_SAMPLE_FROM,
            root_path = self.trainer_config.DATASET.ROOT,
            sigmas =  np_sigmas,
            generate_hms_here = not self.gen_hms_in_mainthread, 
            cv = self.fold,
            cache_data = self.trainer_config.TRAINER.CACHE_DATA,
            num_res_supervisions = self.num_res_supervision,
            debug=self.trainer_config.SAMPLER.DEBUG ,
            original_image_size= self.trainer_config.DATASET.ORIGINAL_IMAGE_SIZE,
            input_size =  self.trainer_config.SAMPLER.INPUT_SIZE,
            hm_lambda_scale = self.trainer_config.MODEL.HM_LAMBDA_SCALE,
            data_augmentation_strategy = self.trainer_config.SAMPLER.DATA_AUG,
            data_augmentation_package = self.trainer_config.SAMPLER.DATA_AUG_PACKAGE,

 
        )

    
        if self.perform_validation:            
            valid_dataset = self.get_evaluation_dataset("validation")
        else:
            valid_dataset = train_dataset
            print("WARNING: NOT performing validation. Instead performing \"validation\" on training set for coord error metrics.")

        #If debug use only main thread to load data bc we only want to show a single plot on screen.
        #If num_workers=0 we are only using the main thread, so persist_workers = False.
        if self.trainer_config.SAMPLER.DEBUG or self.trainer_config.SAMPLER.NUM_WORKERS == 0:
            persist_workers = False
            num_workers_cfg=0
        else:
            persist_workers = True
            num_workers_cfg= self.trainer_config.SAMPLER.NUM_WORKERS    
       

        self.train_dataloader = DataLoader(train_dataset, batch_size=self.data_loader_batch_size, shuffle=True, num_workers=num_workers_cfg, persistent_workers=persist_workers, worker_init_fn=NetworkTrainer.worker_init_fn, pin_memory=True )
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=self.data_loader_batch_size, shuffle=False, num_workers=num_workers_cfg, persistent_workers=persist_workers, worker_init_fn=NetworkTrainer.worker_init_fn, pin_memory=True )
    

    def get_evaluation_dataset(self, split):
        """Gets an evaluation dataset based on split given (must be "validation" or "testing").
            We do not perform patch sampling on evaluation dataset, always returning the full image.
            Patchifying the evaluation image is too large memory constraint to do in batches here.

        Args:
            split (string): Which split of data to return ( "validation" or "testing")

        Returns:
            _type_: Dataset object
        """

        assert split in ["validation", "testing"]

        np_sigmas = [x.cpu().detach().numpy() for x in self.sigmas]
        dataset = DatasetBase(
                annotation_path =self.trainer_config.DATASET.SRC_TARGETS,
                landmarks = self.landmarks,
                LabelGenerator = self.label_generator,
                split = split,
                sample_mode = "full",
                sample_patch_size = self.trainer_config.SAMPLER.PATCH.SAMPLE_PATCH_SIZE,
                sample_patch_bias = self.trainer_config.SAMPLER.PATCH.SAMPLER_BIAS,
                sample_patch_from_resolution = self.trainer_config.SAMPLER.PATCH.RESOLUTION_TO_SAMPLE_FROM,
                root_path = self.trainer_config.DATASET.ROOT,
                sigmas =  np_sigmas,
                generate_hms_here = not self.gen_hms_in_mainthread, 
                cv = self.fold,
                cache_data = self.trainer_config.TRAINER.CACHE_DATA,
                num_res_supervisions = self.num_res_supervision,
                debug=self.trainer_config.SAMPLER.DEBUG,
                data_augmentation_strategy =None,
                original_image_size= self.trainer_config.DATASET.ORIGINAL_IMAGE_SIZE,
                input_size =  self.trainer_config.SAMPLER.INPUT_SIZE,
                hm_lambda_scale = self.trainer_config.MODEL.HM_LAMBDA_SCALE,

            )
        return dataset

    

    @staticmethod
    def worker_init_fn(worker_id):
        imgaug.seed(np.random.get_state()[1][0] + worker_id)

