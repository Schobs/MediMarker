
import enum
import os
import torch
import numpy as np
from time import time
# import multiprocessing as mp
from utils.im_utils.heatmap_manipulation import get_coords
from utils.local_logging.dict_logger import DictLogger
from torch.cuda.amp import GradScaler, autocast
from evaluation.localization_evaluation import success_detection_rate, generate_summary_df
from utils.im_utils.heatmap_manipulation import get_coords
from torchvision.transforms import Resize,InterpolationMode
from torch.utils.data import DataLoader

from abc import ABC, abstractmethod
import imgaug
import copy
import pandas as pd

import matplotlib.pyplot as plt

class NetworkTrainer(ABC):
    """ Super class for trainers. I extend this for trainers for U-Net and PHD-Net. They share some functions.y
    """
    @abstractmethod
    def __init__(self, trainer_config,  is_train= True, dataset_class=None, output_folder=None, comet_logger=None, profiler=None):
        #Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #This is the trainer config dict
        self.trainer_config = trainer_config
        self.is_train = is_train


        #Dataset class to use
        self.dataset_class = dataset_class
        #Dataloader info
        self.data_loader_batch_size = self.trainer_config.SOLVER.DATA_LOADER_BATCH_SIZE
        self.num_batches_per_epoch = self.trainer_config.SOLVER.MINI_BATCH_SIZE
        self.gen_hms_in_mainthread = self.trainer_config.INFERRED_ARGS.GEN_HM_IN_MAINTHREAD
        self.sampler_mode = self.trainer_config.SAMPLER.SAMPLE_MODE
        self.landmarks = self.trainer_config.DATASET.LANDMARKS
        self.training_resolution = self.trainer_config.SAMPLER.PATCH.RESOLUTION_TO_SAMPLE_FROM if self.sampler_mode == "patch" else self.trainer_config.SAMPLER.INPUT_SIZE

       

        #Set up logger & profiler
        self.profiler = profiler
        self.comet_logger = comet_logger
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
        self.dict_logger = None


        #Set up in init of extended class (child)
        self.network = None
        self.train_label_generator = self.eval_label_generator = None
        self.optimizer= None
        self.loss = None
        self.num_res_supervision = None

        #Can be changed in extended class (child)
        self.early_stop_patience = 150
        self.initial_lr = self.trainer_config.SOLVER.BASE_LR

        #Inference params
        self.fit_gauss_inference = self.trainer_config.INFERENCE.FIT_GAUSS

        #Initialize
        self.epoch = 0
        self.best_valid_loss = 999999999999999999999999999
        self.best_valid_coord_error = 999999999999999999999999999
        self.best_valid_loss_epoch = 0
        self.epochs_wo_val_improv = 0
        self.print_initiaization_info = True

    def initialize(self, training_bool=True):
        '''
        Initialize profiler, comet logger, training/val dataloaders, network, optimizer, loss, automixed precision
        and maybe load a checkpoint. 
        
        '''
        # torch.backends.cudnn.benchmark = True

        if self.profiler:
            print("Initialized profiler")
            self.profiler.start()
        
        
        self.initialize_dataloader_settings()
        if training_bool:
            self.set_training_dataloaders()

        self.initialize_network()
        self.initialize_optimizer_and_scheduler()
        self.initialize_loss_function()
        self._maybe_init_amp()

        if self.comet_logger:
            self.comet_logger.log_parameters(self.trainer_config)
        
        #This is the logger that will save epoch results to log & log variables at inference, extend this for any extra stuff you want to log/save at evaluation!
        self.dict_logger = DictLogger(len(self.landmarks), self.regress_sigma, self.loss.loss_seperated_keys, self.dataset_class.additional_sample_attribute_keys)

        self.was_initialized = True

        self.maybe_load_checkpoint()

        self.print_initiaization_info = False


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
            msg = "initialized auto mixed precision."
            # print()
        else:
            msg = "Not initialized auto mixed precision."

        if self.print_initiaization_info:
            print(msg)

    @abstractmethod
    def get_coords_from_heatmap(self, model_output, original_image_size):

        """
        Function to take model output and return coordinates & a Dict of any extra information to log (e.g. max of heatmap)
       """

    def train(self):
        if not self.was_initialized:
            self.initialize(True)

        step = 0
        while self.epoch < self.max_num_epochs:
            

            self.epoch_start_time = time()

            self.network.train()

            generator = iter(self.train_dataloader)

            #We will log the training and validation info here. The keys we set describe all the info we are logging.
            per_epoch_logs =  self.dict_logger.get_epoch_logger()

            print("training")
            # Train for X number of batches per epoch e.g. 250
            for iter_b in range(self.num_batches_per_epoch):
                l, generator = self.run_iteration(generator, self.train_dataloader, backprop=True, split="training", log_coords=False,  logged_vars=per_epoch_logs)
                if self.comet_logger:
                    self.comet_logger.log_metric("training loss iteration", l, step)
                step += 1
            # del generator
            print("validation")

            with torch.no_grad():
                self.network.eval()
                generator = iter(self.valid_dataloader)
                while generator != None:
                    l, generator = self.run_iteration(generator, self.valid_dataloader, backprop=False, split="validation", log_coords=True,  logged_vars=per_epoch_logs)
                 
            self.epoch_end_time = time()
           
            continue_training = self.on_epoch_end(per_epoch_logs)

            if not continue_training:
                if self.profiler:
                    self.profiler.stop()
                break
     
            self.epoch +=1

        #Save the final weights
        if self.comet_logger:
            print("Logging weights as histogram...")
            weights = []
            for name in self.network.named_parameters():
                if 'weight' in name[0]:
                    weights.extend(name[1].detach().cpu().numpy().tolist())
            self.comet_logger.log_histogram_3d(weights, step=self.epoch)



    # @abstractmethod
    # def predict_heatmaps_and_coordinates(self, data_dict):
    #     """ For inference. Predict heatmap and coordinates directly from a data_dict.

    #     Args:
    #         data_dict (_type_): _description_
    #         return_all_layers (bool, optional): _description_. Defaults to False.
    #         resize_to_og (bool, optional): _description_. Defaults to False.
    #     """


    def run_iteration(self, generator, dataloader, backprop, split, log_coords, logged_vars=None, debug=False, direct_data_dict=None):
        so = time()

        #We can either give the generator to be iterated or a data_dict directly
        if direct_data_dict ==  None:
            try:
                data_dict = next(generator)

            except StopIteration:
                if split != "training":
                    return 0, None
                else:
                    generator = iter(dataloader)
                    data_dict = next(generator)
        else:
            data_dict = direct_data_dict


        data =(data_dict['image']).to( self.device )

        # This happens when we regress sigma with >0 workers due to multithreading issues.
        # Currently does not support patch-based, which is raised on run of programme by argument checker.
        if self.gen_hms_in_mainthread:
            data_dict['label'] = self.generate_heatmaps_batch(data_dict, dataloader)          

        #Put targets to device
        target = {key: ([x.to(self.device) for x in val ] if isinstance(val, list) else val.to(self.device) ) for key, val in data_dict['label'].items() }

        
        self.optimizer.zero_grad()

        so = time()
        if self.auto_mixed_precision:
            with autocast():
                output = self.network(data)
                del data
                #Only attempts loss if annotations avaliable for entire batch
                if all(data_dict["annotation_available"]):
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
                    l = torch.tensor(0).to(self.device)
                    loss_dict = {}
         
        else:
            output = self.network(data)
            del data

            #Only attempts loss if annotations avaliable for entire batch
            if all(data_dict["annotation_available"]):
                l, loss_dict = self.loss(output, target, self.sigmas)
            
                if backprop:
                    l.backward()
                    torch.nn.utils.clip_grad_norm_(self.learnable_params, 12)
                    self.optimizer.step() 
                    if self.regress_sigma:
                        self.update_dataloader_sigmas(self.sigmas)
            else:
                l = torch.tensor(0).to(self.device)
                loss_dict = {}

        #Log info from this iteration.
        s= time()
        if list(logged_vars.keys()) != []:
            with torch.no_grad():

                pred_coords, pred_coords_input_size, extra_info, target_coords = self.maybe_get_coords(output, log_coords, data_dict)
            
                logged_vars = self.dict_logger.log_key_variables(logged_vars, pred_coords, extra_info, target_coords, loss_dict, data_dict, log_coords, split)
                if debug:
                    # print("logged_vars ind resyults: ", logged_vars["individual_results"][0]["uid"])
                    # print("data_dict['uid']: ", data_dict['uid'])

                    debug_vars = [x for x in logged_vars["individual_results"]  if x["uid"] in data_dict['uid']]
                    # self, input_dict, prediction_output, predicted_coords
                    self.eval_label_generator.debug_prediction(data_dict, output, pred_coords,pred_coords_input_size, debug_vars, extra_info)
                           
        e = time()
        if self.profiler:
            self.profiler.step()

        del output
        del target
        return l.detach().cpu().numpy(), generator

    

    def maybe_get_coords(self, output, log_coords, data_dict):
        """From output gets coordinates and extra info for logging. If log_coords is false, returns None for all.
            It also decides whether to resize heatmap, rescale coords depending on config settings.

        Args:
            output (_type_): _description_
            log_coords (_type_): _description_
            data_dict (_type_): _description_

        Returns:
            _type_: _description_
        """
        if log_coords:
            pred_coords_input_size, extra_info = self.get_coords_from_heatmap(output, data_dict["original_image_size"])
            pred_coords, target_coords = self.maybe_rescale_coords(pred_coords_input_size, data_dict)                 
        else:
            pred_coords = extra_info = target_coords = pred_coords_input_size = None

        return pred_coords, pred_coords_input_size, extra_info, target_coords

    def on_epoch_end(self, per_epoch_logs):
        """
         Always run to 1000 epochs
        :return:
        """

        new_best_valid = False
        new_best_coord_valid = False

        continue_training = self.epoch < self.max_num_epochs

        #######Logging some end of epoch info #############
        time_taken =self.epoch_end_time - self.epoch_start_time
        per_epoch_logs = self.dict_logger.log_epoch_end_variables(per_epoch_logs, time_taken, self.sigmas, self.optimizer.param_groups[0]['lr'])
      
        #log them else they are lost!
        if self.comet_logger:
            self.dict_logger.log_dict_to_comet(self.comet_logger, per_epoch_logs, self.epoch)

        print("Epoch %s logs: %s" % (self.epoch, per_epoch_logs ))


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

           
    
            
    def maybe_rescale_coords(self, pred_coords, data_dict):
        """Maybe rescale coordinates based on evaluation parameters, and decide which target coords to evaluate against.
            Cases C1:4:
            C1) used full-scale image to train or resized heatmap already: leave predicted, use full-res target
            C2) used low-scale image to train and rescaling coordinates up to full-scale image size 
            C3) use low-scale image to train, want to eval on low-scale coordinates
        Args:
            pred_coords (tensor): coords extracted from output heatmap
            data_dict (dict): dataloader sample dictionary

        Returns:
            tensor, tensor: predicted coordinates and target coordinates for evaluation
        """
        
        #Don't worry in case annotations are not present since these are 0,0 anyway. this is handled elesewhere
        if self.use_full_res_coords:
            target_coords =data_dict['full_res_coords'].to( self.device ) #C1
        else:
            target_coords =np.round(data_dict['target_coords']).to( self.device ) #C3 (and C1 if input size == full res size so full & target the same)
       

        # C2
        if self.use_full_res_coords and not self.resize_first :
            upscale_factor = torch.tensor([data_dict["resizing_factor"][0],data_dict["resizing_factor"][1]]).to(self.device)
            # upscaled_coords = torch.tensor([pred_coords[x]*upscale_factor[x] for x in range(len(pred_coords))]).to(self.device)
            upscaled_coords = torch.mul(pred_coords,upscale_factor)
            pred_coords = torch.round(upscaled_coords)
            # pred_coords = pred_coords * upscale_factor

        return pred_coords, target_coords


    # @abstractmethod 
    def patchify_and_predict(self, single_sample, logged_vars):
        """Function that takens in a large input image, patchifies it and runs each patch through the model & stitches heatmap together

        #1) should split up into patches of given patch-size.
        #2) should run patches through in batches using run_iteration, NOT LOGGING ANYTHING but needs to return the OUTPUTS somehow. 
            MUST ADD OPTION TO RETURN OUTPUTS in run_iteration?
        #3) Need to use method to stitch patches together (future, phdnet will use patch size 512 512 for now).
        #4) call log_key_variables function now with the final big heatmap as the "output". The logging should work as usual from that.
        
        Returns:
            _type_: _description_
        """


    # @abstractmethod 
    def run_inference(self, split, debug=False):
        """ Function to run inference on a full sized input

        #0) instantitate test dataset and dataloader
        #1A) if FULL: 
            i) iterate dataloader and run_iteration each time to go through and save results.
            ii) should run using run_iteration with logged_vars to log
        1B) if PATCHIFYING full_res_output  <- this can be fututre addition
            i) use patchify_and_predict to stitch hm together with logged_vars to log

        #2) need a way to deal with the key dictionary & combine all samples
        #3) need to put evaluation methods in evluation function & import and ues key_dict for analysis
        #4) return individual results & do summary results.
        """

        #If trained using patch, return the full image, else ("full") will return the image size network was trained on. 
        if self.sampler_mode == "patch":
            if self.trainer_config.SAMPLER.PATCH.INFERENCE_MODE == "patchify_and_stitch":
                #In this case we are patchifying the image
                inference_full_image = False
            else:
                #else we are doing it fully_convolutional
                inference_full_image = True
        else:
            #This case is the full sampler
            inference_full_image = True
        inference_resolution = self.training_resolution
        #Load dataloader (Returning coords dont matter, since that's handled in log_key_variables)
        test_dataset = self.get_evaluation_dataset(split, inference_resolution)
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.data_loader_batch_size, shuffle=False, num_workers=self.num_workers_cfg, persistent_workers=self.persist_workers, worker_init_fn=NetworkTrainer.worker_init_fn, pin_memory=True )

        #instantiate interested variables log!
        evaluation_logs =  self.dict_logger.get_evaluation_logger()

        #network evaluation mode
        self.network.eval()

        #then iterate through dataloader and save to log
        generator = iter(self.test_dataloader)
        if inference_full_image:
            while generator != None:
                print("-", end="")
                l, generator = self.run_iteration(generator, self.test_dataloader, backprop=False, split=split, log_coords=True,  logged_vars=evaluation_logs, debug=debug)
            del generator
            print()
        else:
            #this is where we patchify and stitch the input image
            raise NotImplementedError()

        
        summary_results, ind_results = self.evaluation_metrics(evaluation_logs)

 
        return summary_results, ind_results


    def evaluation_metrics(self, evaluation_dict):
        # Success Detection Rate i.e. % images within error thresholds
        radius_list = [1,2,3, 4,5, 6, 7, 8, 9, 10,15,20,25,30,40,50,100]
        outlier_results = {}
        for rad in radius_list:
            out_res_rad = success_detection_rate(evaluation_dict["individual_results"], rad)
            outlier_results[rad] = (out_res_rad)    

        # print("outlier results: ", outlier_results)


        #Generate summary Results
        summary_results = generate_summary_df(evaluation_dict["landmark_errors"], outlier_results )
        ind_results = pd.DataFrame(evaluation_dict["individual_results"] )
    

        return summary_results, ind_results

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
            "sigmas": self.sigmas,
            "training_sampler": self.sampler_mode,
            "training_resolution": self.training_resolution

        }

        if self.amp_grad_scaler is not None:
            state['amp_grad_scaler'] = self.amp_grad_scaler.state_dict()

        torch.save(state, path)

    def run_inference_ensemble_models(self, split, checkpoint_list, smha_choice_idx=0, debug=False):
        
        #If trained using patch, return the full image, else ("full") will return the image size network was trained on. 
        if self.sampler_mode == "patch":
            if self.trainer_config.SAMPLER.PATCH.INFERENCE_MODE == "patchify_and_stitch":
                #In this case we are patchifying the image
                inference_full_image = False
            else:
                #else we are doing it fully_convolutional
                inference_full_image = True
        else:
            #This case is the full sampler
            inference_full_image = True
        inference_resolution = self.training_resolution
        #Load dataloader (Returning coords dont matter, since that's handled in log_key_variables)
        test_dataset = self.get_evaluation_dataset(split, inference_resolution)
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.data_loader_batch_size, shuffle=False, num_workers=0,worker_init_fn=NetworkTrainer.worker_init_fn )

       

        #network evaluation mode
        self.network.eval()

        #then iterate through dataloader and save to log

        #Save ensemble logs, seperated by uncertainty coordinate resolutions
        ensemble_result_dicts = {"smha": [], "emha": [], "ecpv": [], "smha_orig_im_res": [], "emha_orig_im_res": [], "ecpv_orig_im_res": []}

        #Also save just the landmark errors, seperated by uncertainty coordinate resolutions
        all_ind_errors = {
                "smha": [[] for x in range(len(self.landmarks))], 
                "emha": [[] for x in range(len(self.landmarks))], 
                "ecpv": [[] for x in range(len(self.landmarks))],
                "smha_orig_im_res": [[] for x in range(len(self.landmarks))], 
                "emha_orig_im_res": [[] for x in range(len(self.landmarks))], 
                "ecpv_orig_im_res": [[] for x in range(len(self.landmarks))]
                }
        
        
        generator = iter(self.test_dataloader)
        if inference_full_image:
            # Need to load and get results from each checkpoint. Load checkpoint for each batch because of memory issues running through entire dataloader
            # and saving multiple outputs for every checkpoint. In future can improve this by going through X (e.g.200 samples/10 batches) before changing checkpoint.
            while generator != None:
                try:
                    evaluation_logs =  self.dict_logger.ensemble_inference_log_template()
                    direct_data_dict = next(generator)
                    # print("direct data dict: ", direct_data_dict)
                    for ckpt in checkpoint_list:
                        self.load_checkpoint(ckpt, training_bool=False)
                        # The logger will not try and save the error since annotation_available is false.
                        # We need to save the entire heatmap as extra information.
                        #directly pass the next data_dict to run_iteration rather than iterate it within.
                        l, _ = self.run_iteration(generator, self.test_dataloader, backprop=False, split=split, log_coords=True,  logged_vars=evaluation_logs, debug=debug, direct_data_dict=direct_data_dict)
                        # print("logged vars: ", len(evaluation_logs["individual_results"]), evaluation_logs["individual_results"][0].keys())

                    #Analyse batch for s-mha, e-mha, and e-cpv and maybe errors (if we have annotations)
                    # ensemble_uncertainty_dict = analyse_ensemble_uncertainty(evaluation_logs, smha_choice_idx=smha_choice_idx)
                    ensembles_analyzed, ind_landmark_errors = self.ensemble_inference_postprocessing(evaluation_logs, smha_choice_idx)

                    # print("ensembles analyzed keys: ", ensembles_analyzed.keys())

                    for k_ in list(ensemble_result_dicts.keys()):
                        ensemble_result_dicts[k_].extend(ensembles_analyzed[k_])

                    # print("ensemble_result_dicts", ensemble_result_dicts)
                        
                    for ens_key, coord_extact_methods in ind_landmark_errors.items():
                        for ile_idx, ind_lm_ers in enumerate(coord_extact_methods):
                            # print(ens_key, ile_idx, (ind_lm_ers))
                            all_ind_errors[ens_key][ile_idx].extend(ind_lm_ers)

                    #Now we have the results for all checkpoints for this batch, we can save the results to disk.
                    # save_ensemble_inference_spreadsheet(ensembles_analyzed)



                except StopIteration:
                    generator = None
                print("-", end="")      
            print("no more in generator")           
            del generator

            # return 
            # #now save
            # save_ensemble_inference_spreadsheet(ensembles_analyzed)


            # print()
        else:
            #this is where we patchify and stitch the input image
            raise NotImplementedError()


        # Success Detection Rate i.e. % images within error thresholds
        radius_list = [1,2,3, 4,5, 6, 7, 8, 9, 10,15,20,25,30,40,50,100]

        #save SDR for each of the uncertainty coordinate extraction techniques
        all_outlier_results = {}
        #first get the keys (e.g. s-mha, e-mha, e-cpv)
        uncert_keys =  list(ind_landmark_errors.keys())
        # print("uncert keys: ", uncert_keys)

        all_summary_results = {}
        for u_key in uncert_keys:
            outlier_results = {}
            # print("len ensemble_result_dicts all_", u_key , len(ensemble_result_dicts[u_key]) )

            for rad in radius_list:
                out_res_rad = success_detection_rate(ensemble_result_dicts[u_key], rad)
                outlier_results[rad] = (out_res_rad)    
            all_outlier_results[u_key] = outlier_results


        # print("outlier results: ", outlier_results)

            #Generate summary Results
            summary_results = generate_summary_df(all_ind_errors[u_key], outlier_results)
            all_summary_results[u_key] = summary_results
        
        ind_results = {}
        for k_, v_ in ensemble_result_dicts.items():
            ind_results[k_] = pd.DataFrame(v_)
        
        # [pd.DataFrame(x) for x_key, x in ensemble_result_dicts]
        #     summary_results, ind_results = self.evaluation_metrics(evaluation_logs)
        return all_summary_results, ind_results
            # return summary_results, ind_results


    

    def ensemble_inference_postprocessing(self, evaluation_logs, smha_choice_idx):
        """ Analyze ensemble results to generate s-mha, e-mha and e-cpv.

        Args:
            evaluation_logs ([dict[]): list of dicts of predicitons. each uid has a list of individual_dict from each checkpoint.
                Keys in each individual_dict result:
                ['annotation_available', 'predicted_coords', 'uid', 'image_path', 'Error All Mean', 'Error All Std', 'ind_errors', 
                'target_coords', 'L0', 'L1', 'L2', 'L3', 'hm_max', 'coords_og_size', 'final_heatmaps']
        """
        #Get the large 2D list of landmark errors if available


        #Combine the multiple ensemble predictions, using uid as key
        sorted_dict_by_uid = {}
        for ind_res in evaluation_logs["individual_results"]:
            uid = ind_res["uid"]
            if uid not in sorted_dict_by_uid:
                sorted_dict_by_uid[uid] = {}
                sorted_dict_by_uid[uid]["ind_preds"] = []
            sorted_dict_by_uid[uid]["ind_preds"].append(ind_res)

        all_ensemble_results_dicts= {"smha": [], "emha": [], "ecpv": [],  "smha_orig_im_res": [],  "emha_orig_im_res": [],  "ecpv_orig_im_res": []}
        ind_errors = {
                "smha": [[] for x in range(len(self.landmarks))], 
                "emha": [[] for x in range(len(self.landmarks))], 
                "ecpv": [[] for x in range(len(self.landmarks))],
                "smha_orig_im_res": [[] for x in range(len(self.landmarks))], 
                "emha_orig_im_res": [[] for x in range(len(self.landmarks))], 
                "ecpv_orig_im_res": [[] for x in range(len(self.landmarks))]
                }
        
        #go through each sample (uid) and get the ensemble results
        for uid_, results_dict_list in sorted_dict_by_uid.items(): 
            
            #Get all the individual landmark results for this sample for all models in the ensemble
            individual_results = results_dict_list["ind_preds"]
            ensemble_results_dict = {}

            # Log standard descriptive info for each sample e.g. uid, image_path, annotation_available etc
            for k_ in list(all_ensemble_results_dicts.keys()):
                ensemble_results_dict[k_] = {}
                for sample_info_key in evaluation_logs["sample_info_log_keys"]:
                    ensemble_results_dict[k_][sample_info_key] = individual_results[0][sample_info_key]


            if individual_results[0]["annotation_available"]:
                calc_error = True
            else:
                calc_error = False

            #assert that target coords are the same from all models i.e. the uid is matching for all.
            if calc_error:
                assert all([y.all()==True for y in [(ind_res["target_coords"] == individual_results[0]["target_coords"]) for ind_res in individual_results]]), "Target coords are not the same for all models in the ensemble"
                target_coords = individual_results[0]["target_coords"]
                target_coords_original_resolution = individual_results[0]["target_coords_original_resolution"]

            ################## 1) S-MHA ###########################
                
            ensemble_results_dict["smha"]["predicted_coords"] = individual_results[smha_choice_idx]["predicted_coords"]
            ensemble_results_dict["smha"]["uncertainty"] = individual_results[smha_choice_idx]["hm_max"]
            ensemble_results_dict["smha_orig_im_res"]["predicted_coords"] = individual_results[smha_choice_idx]["predicted_coords_original_resolution"]       
            ensemble_results_dict["smha_orig_im_res"]["uncertainty"] = individual_results[smha_choice_idx]["hm_max"]

            #Save inference resolution results for each landmark individually
            for c_idx, pred_coord_t in enumerate(individual_results[smha_choice_idx]["predicted_coords"]):
                ensemble_results_dict["smha"]["predicted_L"+str(c_idx)] = pred_coord_t
                ensemble_results_dict["smha"]["uncertainty_L"+str(c_idx)] = ensemble_results_dict["smha"]["uncertainty"][c_idx][0]

            #Save original image resolution results for each landmark individually
            for c_idx, pred_coord_t in enumerate(individual_results[smha_choice_idx]["predicted_coords_original_resolution"]):
                ensemble_results_dict["smha_orig_im_res"]["predicted_L"+str(c_idx)] = pred_coord_t
                ensemble_results_dict["smha_orig_im_res"]["uncertainty_L"+str(c_idx)] = ensemble_results_dict["smha_orig_im_res"]["uncertainty"][c_idx][0]

            ####Calculate the error if annotation available
            if calc_error:
                #Error for inference resolution
                ensemble_results_dict["smha"]["ind_errors"] = individual_results[smha_choice_idx]["ind_errors"]
                ensemble_results_dict["smha"]["Error All Mean"] = np.mean(individual_results[smha_choice_idx]["ind_errors"])
                ensemble_results_dict["smha"]["Error All Std"] = np.std(individual_results[smha_choice_idx]["ind_errors"])

                for cidx, p_coord in enumerate(individual_results[smha_choice_idx]["ind_errors"]):
                    ensemble_results_dict["smha"]["L"+str(cidx)] = p_coord
                    ensemble_results_dict["smha"]["uncertainty L"+str(cidx)] = individual_results[smha_choice_idx]["hm_max"][cidx]
                    ind_errors["smha"][cidx].append(p_coord)

                #Error for original image resolution
                ensemble_results_dict["smha_orig_im_res"]["ind_errors"] = individual_results[smha_choice_idx]["ind_errors (Original Resolution)"]
                ensemble_results_dict["smha_orig_im_res"]["Error All Mean"] = np.mean(individual_results[smha_choice_idx]["ind_errors (Original Resolution)"])
                ensemble_results_dict["smha_orig_im_res"]["Error All Std"] = np.std(individual_results[smha_choice_idx]["ind_errors (Original Resolution)"])

                for cidx, p_coord in enumerate(individual_results[smha_choice_idx]["ind_errors (Original Resolution)"]):
                    ensemble_results_dict["smha_orig_im_res"]["L"+str(cidx)] = p_coord
                    # ensemble_results_dict["smha"]["uncertainty L"+str(cidx)] = individual_results[smha_choice_idx]["hm_max"][cidx]
                    ind_errors["smha_orig_im_res"][cidx].append(p_coord)

           
            ################## 2) E_CPV ###########################

            #Calculate the E-CPV
            average_coords = np.mean([dic['predicted_coords'] for dic in individual_results] , axis=0)
            average_coords_og_resolution = np.mean([dic['predicted_coords_original_resolution'] for dic in individual_results] , axis=0)

            #the actual e-cpv uncertainty measure should be the same for both inference and resized to maintain the same scaling between all images.
            all_coord_vars = []
            for coord_idx, coord in enumerate(average_coords):
                all_coord_vars.append(np.mean([abs(np.linalg.norm(coord- x)) for x in [dic['predicted_coords'][coord_idx] for dic in individual_results]]))
                    

            ensemble_results_dict["ecpv"]["predicted_coords"] = np.round(average_coords)
            ensemble_results_dict["ecpv"]["uncertainty"] = all_coord_vars
            ensemble_results_dict["ecpv_orig_im_res"]["predicted_coords"] = np.round(average_coords_og_resolution)
            ensemble_results_dict["ecpv_orig_im_res"]["uncertainty"] = all_coord_vars

            #Save inference resolution results for each landmark individually
            for c_idx, pred_coord_t in enumerate(ensemble_results_dict["ecpv"]["predicted_coords"]):
                ensemble_results_dict["ecpv"]["predicted_L"+str(c_idx)] = pred_coord_t
                ensemble_results_dict["ecpv"]["uncertainty_L"+str(c_idx)] = ensemble_results_dict["ecpv"]["uncertainty"][c_idx]

            #Save original image resolution results for each landmark individually
            for c_idx, pred_coord_t in enumerate(ensemble_results_dict["ecpv_orig_im_res"]["predicted_coords"]):
                ensemble_results_dict["ecpv_orig_im_res"]["predicted_L"+str(c_idx)] = pred_coord_t
                ensemble_results_dict["ecpv_orig_im_res"]["uncertainty_L"+str(c_idx)] = ensemble_results_dict["ecpv_orig_im_res"]["uncertainty"][c_idx]

            ####Calculate the error if annotation available
            if calc_error:
                #Error for inference resolution
                ecpv_errors = np.linalg.norm(ensemble_results_dict["ecpv"]["predicted_coords"] - target_coords, axis=1)
                ensemble_results_dict["ecpv"]["ind_errors"] =ecpv_errors
                ensemble_results_dict["ecpv"]["Error All Mean"] = np.mean(ecpv_errors)
                ensemble_results_dict["ecpv"]["Error All Std"] = np.std(ecpv_errors)

                for cidx, p_coord in enumerate(ecpv_errors):
                    ensemble_results_dict["ecpv"]["L"+str(cidx)] = p_coord[0]
                    ensemble_results_dict["ecpv"]["uncertainty L"+str(cidx)] = all_coord_vars[cidx]
                    ind_errors["ecpv"][cidx].append(p_coord)


                #Error for original image resolution
                ecpv_errors_og_res = np.linalg.norm(ensemble_results_dict["ecpv_orig_im_res"]["predicted_coords"] - target_coords_original_resolution, axis=1)
                ensemble_results_dict["ecpv_orig_im_res"]["ind_errors"] =ecpv_errors_og_res
                ensemble_results_dict["ecpv_orig_im_res"]["Error All Mean"] = np.mean(ecpv_errors_og_res)
                ensemble_results_dict["ecpv_orig_im_res"]["Error All Std"] = np.std(ecpv_errors_og_res)

                for cidx, p_coord in enumerate(ecpv_errors_og_res):
                    ensemble_results_dict["ecpv_orig_im_res"]["L"+str(cidx)] = p_coord[0]
                    # ensemble_results_dict["ecpv"]["uncertainty L"+str(cidx)+"_orig_im_res"] = all_coord_vars[cidx]
                    ind_errors["ecpv_orig_im_res"][cidx].append(p_coord)
            
            ################## 3) E_MHA ###########################

            # 3.1) Average all the heatmaps
            all_ind_heatmaps = [dic['final_heatmaps'] for dic in individual_results]
            #Create an empty map and add all maps to it, then average
            ensemble_map = torch.zeros((1, all_ind_heatmaps[0].shape[0], all_ind_heatmaps[0].shape[1], all_ind_heatmaps[0].shape[2]))
            
            for model_idx, per_model_hms in enumerate(all_ind_heatmaps):
                #Make sure all the heatmaps are the same size
                assert all([x.shape == per_model_hms[0].shape for x in per_model_hms]), "Output ensemble heatmaps are not the same size!"
                for l_idx in range(len(per_model_hms)):
                    ensemble_map[0,l_idx] += per_model_hms[l_idx]
            ensemble_map[0] /= len(all_ind_heatmaps)
       

            # 3.2) Extract the coords from the averaged heatmaps
            pred_coords, max_values = get_coords(ensemble_map)
            #resize the coords to the original image resolution as well
            pred_coords_original_resolution = np.round(((pred_coords.detach().cpu().numpy())) * individual_results[0]["resizing_factor"])[0]
            
            pred_coords =  np.round(pred_coords).cpu().detach().numpy()[0]
            max_values = max_values.cpu().detach().numpy()[0]
          
            ensemble_results_dict["emha"]["predicted_coords"] = pred_coords
            ensemble_results_dict["emha"]["uncertainty"] = max_values

            ensemble_results_dict["emha_orig_im_res"]["predicted_coords"] = pred_coords_original_resolution
            ensemble_results_dict["emha_orig_im_res"]["uncertainty"] = max_values



            #Save inference resolution results for each landmark individually
            for c_idx, pred_coord_t in enumerate(ensemble_results_dict["emha"]["predicted_coords"]):
                ensemble_results_dict["emha"]["predicted_L"+str(c_idx)] = pred_coord_t
                ensemble_results_dict["emha"]["uncertainty_L"+str(c_idx)] = ensemble_results_dict["emha"]["uncertainty"][c_idx][0]
            
            #Save original image resolution results for each landmark individually
            for c_idx, pred_coord_t in enumerate(ensemble_results_dict["emha_orig_im_res"]["predicted_coords"]):
                ensemble_results_dict["emha_orig_im_res"]["predicted_L"+str(c_idx)] = pred_coord_t
                ensemble_results_dict["emha_orig_im_res"]["uncertainty_L"+str(c_idx)] = ensemble_results_dict["emha_orig_im_res"]["uncertainty"][c_idx][0]

            if calc_error:
                #Error for inference resolution
                emha_errors = np.linalg.norm(ensemble_results_dict["emha"]["predicted_coords"] - target_coords, axis=1)
                ensemble_results_dict["emha"]["ind_errors"] = emha_errors
                ensemble_results_dict["emha"]["Error All Mean"] = np.mean(emha_errors)
                ensemble_results_dict["emha"]["Error All Std"] = np.std(emha_errors)

                for cidx, p_coord in enumerate(emha_errors):
                    ensemble_results_dict["emha"]["L"+str(cidx)] = p_coord
                    ensemble_results_dict["emha"]["uncertainty L"+str(cidx)] = ensemble_results_dict["emha"]["uncertainty"][cidx]
                    ind_errors["emha"][cidx].append(p_coord)


                #Error for original image resolution
                emha_errors_og_res = np.linalg.norm(ensemble_results_dict["emha_orig_im_res"]["predicted_coords"] - target_coords_original_resolution, axis=1)
                ensemble_results_dict["emha_orig_im_res"]["ind_errors"] = emha_errors_og_res
                ensemble_results_dict["emha_orig_im_res"]["Error All Mean"] = np.mean(emha_errors_og_res)
                ensemble_results_dict["emha_orig_im_res"]["Error All Std"] = np.std(emha_errors_og_res)

                for cidx, p_coord in enumerate(emha_errors_og_res):
                    ensemble_results_dict["emha_orig_im_res"]["L"+str(cidx)] = p_coord[0]
                    ind_errors["emha_orig_im_res"][cidx].append(p_coord)

            for k_ in list(all_ensemble_results_dicts.keys()):
                all_ensemble_results_dicts[k_].append(ensemble_results_dict[k_])


        return all_ensemble_results_dicts, ind_errors

         


    def maybe_load_checkpoint(self):
        if self.continue_checkpoint:
            self.load_checkpoint(self.continue_checkpoint, self.is_train)
    
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

        if training_bool:
            self.best_valid_loss = checkpoint_info['best_valid_loss']
            self.best_valid_loss_epoch = checkpoint_info['best_valid_loss_epoch']
            self.best_valid_coord_error = checkpoint_info['best_valid_coord_error']
            self.best_valid_coords_epoch = checkpoint_info["best_valid_coords_epoch"]
            self.epochs_wo_val_improv = checkpoint_info["epochs_wo_improvement"]

            # self.best_valid_loss = 0
            # self.best_valid_loss_epoch = 0
            # self.best_valid_coord_error = 0
            # self.best_valid_coords_epoch = 0
            # self.epochs_wo_val_improv = 0
        
        #Allow legacy models to be loaded (they didn't use to save sigmas)
        if "sigmas" in checkpoint_info:
            self.sigmas= checkpoint_info["sigmas"]
        
        #if not saved, default to full since this was the only option for legacy models
        if "training_sampler" in checkpoint_info:
            self.sampler_mode = checkpoint_info["training_sampler"]
        else:
            self.sampler_mode = "full"

        #if not saved, default to input_size since this was the only option for legacy models
        if "training_resolution" in checkpoint_info:
            self.training_resolution = checkpoint_info["training_resolution"]
        else:
            self.training_resolution = self.trainer_config.SAMPLER.INPUT_SIZE

        self.checkpoint_loading_checking()
      
        if self.auto_mixed_precision:
            self._maybe_init_amp()

            if 'amp_grad_scaler' in checkpoint_info.keys():
                self.amp_grad_scaler.load_state_dict(checkpoint_info['amp_grad_scaler'])

        if self.print_initiaization_info:
            print("Loaded checkpoint %s. Epoch: %s, " % (model_path, self.epoch ))

    def checkpoint_loading_checking(self):
        """Checks that the loaded checkpoint is compatible with the current model and training settings.

        Raises:
            ValueError: _description_
            ValueError: _description_
        """

        #Check the sampler in config is the same as the one in the checkpoint.
        if self.sampler_mode != self.trainer_config.SAMPLER.SAMPLE_MODE:
            raise ValueError("model was trained using SAMPLER.SAMPLE_MODE %s but attempting to load with SAMPLER.SAMPLE_MODE %s. \
                Please amend this in config file." % (self.sampler_mode, self.trainer_config.SAMPLER.SAMPLE_MODE))

        #check if the training resolution from config is the same as the one in the checkpoint.
        if self.sampler_mode == "patch":
            if self.training_resolution != self.trainer_config.SAMPLER.PATCH.RESOLUTION_TO_SAMPLE_FROM:
                raise ValueError("model was trained using SAMPLER.PATCH.RESOLUTION_TO_SAMPLE_FROM %s but attempting to load with self.training_resolution %s. \
                    Please amend this in config file." % (self.training_resolution, self.training_resolution))
        else:
            if self.training_resolution != self.trainer_config.SAMPLER.INPUT_SIZE:
                raise ValueError("model was trained using SAMPLER.INPUT_SIZE %s but attempting to load with self.training_resolution %s. \
                    Please amend this in config file." % (self.training_resolution, self.trainer_config.SAMPLER.INPUT_SIZE))


    def initialize_dataloader_settings(self):
        """Initializes dataloader settings. If debug use only main thread to load data bc we only 
            want to show a single plot on screen. 
            If num_workers=0 we are only using the main thread, so persist_workers = False.
        """
       
        if self.trainer_config.SAMPLER.DEBUG or self.trainer_config.SAMPLER.NUM_WORKERS == 0:
            self.persist_workers = False
            self.num_workers_cfg=0
        else:
            self.persist_workers = True
            self.num_workers_cfg= self.trainer_config.SAMPLER.NUM_WORKERS    

    def set_training_dataloaders(self):
        """
        set train_dataset, valid_dataset and train_dataloader and valid_dataloader here.
        """

        np_sigmas = [x.cpu().detach().numpy() for x in self.sigmas]
    
        train_dataset = self.dataset_class(
            annotation_path =self.trainer_config.DATASET.SRC_TARGETS,
            landmarks = self.landmarks,
            LabelGenerator = self.train_label_generator,
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
            input_size =  self.trainer_config.SAMPLER.INPUT_SIZE,
            hm_lambda_scale = self.trainer_config.MODEL.HM_LAMBDA_SCALE,
            data_augmentation_strategy = self.trainer_config.SAMPLER.DATA_AUG,
            data_augmentation_package = self.trainer_config.SAMPLER.DATA_AUG_PACKAGE,
            dataset_split_size = self.trainer_config.DATASET.TRAINSET_SIZE,
        )
    
        if self.perform_validation:
            #if patchify, we want to return the full image
            if self.sampler_mode == "patch":            
                valid_dataset = self.get_evaluation_dataset("validation", self.trainer_config.SAMPLER.PATCH.RESOLUTION_TO_SAMPLE_FROM)
            else:
                valid_dataset = self.get_evaluation_dataset("validation", self.trainer_config.SAMPLER.INPUT_SIZE)

        else:
            if self.sampler_mode == "patch":            
                valid_dataset = self.get_evaluation_dataset("training", self.trainer_config.SAMPLER.PATCH.RESOLUTION_TO_SAMPLE_FROM, dataset_split_size=self.trainer_config.DATASET.TRAINSET_SIZE)
            else:
                valid_dataset = self.get_evaluation_dataset("training", self.trainer_config.SAMPLER.INPUT_SIZE, dataset_split_size=self.trainer_config.DATASET.TRAINSET_SIZE)
            print("WARNING: NOT performing validation. Instead performing \"validation\" on training set for coord error metrics.")

        print("Using %s Dataloader workers and persist workers bool : %s " % (self.num_workers_cfg, self.persist_workers))
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.data_loader_batch_size, shuffle=True, num_workers=self.num_workers_cfg, persistent_workers=self.persist_workers, worker_init_fn=NetworkTrainer.worker_init_fn, pin_memory=True )
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=self.data_loader_batch_size, shuffle=False, num_workers=self.num_workers_cfg, persistent_workers=self.persist_workers, worker_init_fn=NetworkTrainer.worker_init_fn, pin_memory=True )
    

    def get_evaluation_dataset(self, split, load_im_size, dataset_split_size=-1):
        """Gets an evaluation dataset based on split given (must be "validation" or "testing").
            We do not perform patch sampling on evaluation dataset, always returning the full image (sample_mode = "full").
            Patchifying the evaluation image is too large memory constraint to do in batches here.

        Args:
            split (string): Which split of data to return ( "validation" or "testing")

        Returns:
            _type_: Dataset object
        """

        # assert split in ["validation", "testing"]
        np_sigmas = [x.cpu().detach().numpy() for x in self.sigmas]
        dataset = self.dataset_class(
                annotation_path =self.trainer_config.DATASET.SRC_TARGETS,
                landmarks = self.landmarks,
                LabelGenerator = self.eval_label_generator,
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
                input_size =  load_im_size,
                hm_lambda_scale = self.trainer_config.MODEL.HM_LAMBDA_SCALE,
                dataset_split_size= dataset_split_size

            )
        return dataset

    def generate_heatmaps_batch(self, data_dict, dataloader):
        batch_hms = []
        np_sigmas = [x.cpu().detach().numpy() for x in self.sigmas]
        b_= [dataloader.dataset.generate_labels(x, np_sigmas) for x in data_dict["target_coords"]]
        for x in b_:
            if batch_hms == []:
                batch_hms = [[y] for y in x]
            else:
                for hm_idx, hm in enumerate(x):
                    batch_hms[hm_idx].append(hm)
        batch_hms = [torch.stack(x) for x in batch_hms]

        return batch_hms

    @staticmethod
    def worker_init_fn(worker_id):
        imgaug.seed(np.random.get_state()[1][0] + worker_id)

