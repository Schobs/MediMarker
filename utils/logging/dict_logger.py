import torch
import numpy as np
import copy

class DictLogger():
    """ A dictionary based logger to save results. Extend this class to log any extra variables!
    """
    def __init__(self, num_landmarks, is_regressing_sigma, multi_part_loss_keys, additional_sample_attribute_keys, log_valid_heatmap=False, log_inference_heatmap=False, log_fitted_gauss=False, log_inference_heatmap_wo_like=False, model_type="default"):
        #Device

        self.num_landmarks = num_landmarks
        self.is_regressing_sigma = is_regressing_sigma
        self.is_log_valid_heatmap = log_valid_heatmap
        self.is_log_inference_heatmap = log_inference_heatmap
        self.log_inference_heatmap_wo_like = log_inference_heatmap_wo_like
        self.multi_part_loss_keys = multi_part_loss_keys
        self.add_sample_att_keys = additional_sample_attribute_keys
        self.standard_info_keys = ["uid", "full_res_coords", "annotation_available", "image_path", "target_coords",  "resizing_factor", "original_image_size"] 
        self.log_fitted_gauss = log_fitted_gauss
        self.model_type = model_type
        self.per_epoch_logs = self.per_epoch_log_template()
        self.evaluation_logged_vars = self.evaluation_log_template(model_type)
        self.ensemble_inference_logs = self.ensemble_inference_log_template()
        
        #also add the additional sample attributes to the standard info keys.
        self.standard_info_keys = self.standard_info_keys + self.add_sample_att_keys

    def per_epoch_log_template(self):
        logged_per_epoch =  {"valid_coord_error_mean": [], "epoch_time": [], "lr": [], "individual_results_extra_keys": []}
        if self.is_log_valid_heatmap:
            logged_per_epoch["individual_results_extra_keys"] = ["final_heatmaps"] 
            logged_per_epoch["individual_results"] = []
            logged_per_epoch["final_heatmaps"] = []

        if self.is_regressing_sigma:
            logged_per_epoch["sigmas_mean"] =  []
            for sig in range(self.num_landmarks):
                 logged_per_epoch["sigma_"+str(sig)] = []


        #initialise keys for logging the multi-part losses.
        for key_ in self.multi_part_loss_keys:
            logged_per_epoch["training_" + key_] = []
            logged_per_epoch["validation_" + key_] = []

        return logged_per_epoch

    def evaluation_log_template(self, model_type):
        # if self.model_type == "default":       

        eval_logs = {"individual_results": [], "landmark_errors": [[] for x in range(self.num_landmarks)],
            "landmark_errors_original_resolution": [[] for x in range(self.num_landmarks)],
            "sample_info_log_keys": self.standard_info_keys, "individual_results_extra_keys": ['hm_max', 'pred_coords_input_size', 'target_coords_input_size']}

        if self.is_log_inference_heatmap:
            eval_logs["individual_results_extra_keys"].append("final_heatmaps")
            

        if self.log_fitted_gauss:
            eval_logs["individual_results_extra_keys"].append("fitted_gauss")

        if model_type == "gp":       
            if self.log_inference_heatmap_wo_like:
                    eval_logs["individual_results_extra_keys"].append("final_heatmaps_wo_like_noise")
            eval_logs["individual_results_extra_keys"].extend(['kernel_cov_matr','likelihood_noise','full_cov_matrix'])
            

        return eval_logs
    

    #     return eval_logs

    def ensemble_inference_log_template(self):


        # print("standard_info_keys", standard_info_keys)
        return {"individual_results": [], "landmark_errors": [[] for x in range(self.num_landmarks)], 
        "landmark_errors_original_resolution": [[] for x in range(self.num_landmarks)],
        "sample_info_log_keys": self.standard_info_keys, "individual_results_extra_keys": ['final_heatmaps', 'hm_max', 'coords_og_size']}


    
    def get_epoch_logger(self):
        return copy.deepcopy(self.per_epoch_logs)

    def get_evaluation_logger(self):
        return copy.deepcopy(self.evaluation_logged_vars)

    
    def get_ensemble_inference_logger(self):
        return copy.deepcopy(self.ensemble_inference_log_template)

    def log_key_variables(self, log_dict, pred_coords, extra_info, target_coords, loss_dict, data_dict, log_coords, split):
        """Logs base key variables. Should be extended by child class for non-generic logging variables.

        Args:
            output (_type_): _description_
            loss (_type_): _description_
            data_dict (_type_): _description_
            logged_vars (_type_): _description_
        """
        #1) Log training/validation losses based on split.
        vars_to_log = list(log_dict.keys())
        for key, value in loss_dict.items():
            key_ = split+ "_" + key

            if key_ in vars_to_log:
                log_dict[key_].append(value)

 
        #Only log info we requested in the evaluation/ensemble templates
        if extra_info is not None:
            extra_info = {k: extra_info[k] for k in log_dict['individual_results_extra_keys'] if k in extra_info}

        if log_coords:

            

            #Get coord error of the input resolution to network
            if torch.is_tensor(pred_coords):
                coord_error = torch.linalg.norm((pred_coords- target_coords), axis=2)
            else:
                coord_error = np.linalg.norm((pred_coords- target_coords), axis=2)


      
    
            if split == "validation":
                if "valid_coord_error_mean"  in vars_to_log:
                    log_dict["valid_coord_error_mean" ].append(np.mean(coord_error.detach().cpu().numpy()))
            elif split == "training":
                if "train_coord_error_mean" in vars_to_log:
                    log_dict["train_coord_error_mean" ].append(np.mean(coord_error.detach().cpu().numpy()))

            #Save data for ecah sample individually
            if "individual_results" in vars_to_log:

                for idx in range(len(pred_coords)):
                    ind_dict = {}

                    #First log standard info about the sample, maybe detaching if it is a tensor
                    for standard_info_key in self.standard_info_keys:

                        # print("printing log: ",standard_info_key, data_dict[standard_info_key], standard_info_key)
                        if isinstance(data_dict[standard_info_key], dict):
                            for k, v in data_dict[standard_info_key].items():
                                if torch.is_tensor(v):
                                    v = v[idx].detach().cpu().numpy()
                                ind_dict[standard_info_key+"_"+k] = v
                        else:

                            data_point = data_dict[standard_info_key][idx]

                            if torch.is_tensor(data_point):
                                data_point = data_point.detach().cpu().numpy()

                            ind_dict[standard_info_key] = data_point


                    ind_dict["predicted_coords"] = ((pred_coords[idx].detach().cpu().numpy()))

           

                    #If target annotation not avaliable, we don't know the error
                    if ind_dict["annotation_available"] == False:

                        #Save for network input resolution
                        ind_dict["Error All Mean"] = None
                        ind_dict["Error All Std"] = None
                        ind_dict["ind_errors"] = None
                        ind_dict["target_coords"] = None

                        for coord_idx, er in enumerate(coord_error[idx]):
                            ind_dict["L"+str(coord_idx)] = None


                            
                    else:
                        #Save for network input resolution
                        ind_dict["Error All Mean"] = (np.mean(coord_error[idx].detach().cpu().numpy()))
                        ind_dict["Error All Std"] = (np.std(coord_error[idx].detach().cpu().numpy()))
                        ind_dict["ind_errors"] = ((coord_error[idx].detach().cpu().numpy()))
                        ind_dict["target_coords"] = ((target_coords[idx].detach().cpu().numpy()))


                        for coord_idx, er in enumerate(coord_error[idx]):
                            ind_dict["L"+str(coord_idx)] = er.detach().cpu().numpy()

                            if "landmark_errors" in vars_to_log:
                                log_dict["landmark_errors"][coord_idx].append(er.detach().cpu().numpy())


                    #any extra info returned by the child class when calculating coords from outputs e.g. heatmap_max
                    for key_ in list(extra_info.keys()):
                       
                        if "debug" not in key_:

                            if torch.is_tensor(extra_info[key_]):
                                ind_dict[key_] = ((extra_info[key_][idx].detach().cpu().numpy()))
                            else:
                                ind_dict[key_] = ((extra_info[key_][idx]))
                            
                    log_dict["individual_results"].append(ind_dict)
     
        
        return log_dict

    def log_epoch_end_variables(self, per_epoch_logs, time, sigmas, learning_rate ):
        """Logs end of epoch variables. If given a list of things to log it generates the mean of the list.

        Args:
            per_epoch_logs (Dict): Dict of variables to log.
            time (float): time it took for epoch
            sigmas ([Tensor]): Sigmas for the heatmap.
        """
        if "lr" in list(per_epoch_logs.keys()):
            per_epoch_logs["lr"] =  learning_rate
        if "epoch_time" in list(per_epoch_logs.keys()):
            per_epoch_logs["epoch_time"] =  time
        if "sigmas_mean" in list(per_epoch_logs.keys()):
            np_sigmas = [x.cpu().detach().numpy() for x in sigmas]
            per_epoch_logs["sigmas_mean"] = (np.mean(np_sigmas))

            for idx, sig in enumerate(np_sigmas):
                if "sigma_"+str(idx) in list(per_epoch_logs.keys()):
                    per_epoch_logs["sigma_"+str(idx)] = sig

        for key, value in per_epoch_logs.items():
            
            #Handle heatmap logging
            if key == "individual_results" and "final_heatmaps" in list(per_epoch_logs.keys()):
                [per_epoch_logs["final_heatmaps"].append([x["uid"]+"_training_phase", x["final_heatmaps"]]) for x in value]
            elif key == "individual_results" and "final_heatmaps" in list(per_epoch_logs.keys()):
                [per_epoch_logs["final_heatmaps_wo_like_noise"].append([x["uid"]+"_training_phase_nolikenoise", x["final_heatmaps_wo_like_noise"]]) for x in value]
            else:
                #Don't worry about averaging these, we handled those above.
                if key in ["individual_results_extra_keys", "final_heatmaps", "final_heatmaps_wo_like_noise"]:
                    continue
                #get the mean of all the batches from the training/validations. 
                if isinstance(value, list):
                    per_epoch_logs[key] = np.round(np.mean([x.detach().cpu().numpy() if torch.is_tensor(x) else x for x in value]),5)
                if torch.is_tensor(value):
                    per_epoch_logs[key] = np.round(value.detach().cpu().numpy(), 5)
            
        return per_epoch_logs
    

    def log_dict_to_comet(self, comet_logger, dict_to_log, time_step):
        for key, value in dict_to_log.items():
            #Don't log this meta info
            if key in ["individual_results_extra_keys", "individual_results"]:
                continue 
            #Log heatmaps during training to keep track.
            if key == "final_heatmaps" or key == "final_heatmaps_wo_like_noise":
                for uid, heatmaps in value:
                    for l_idx, heatmap in enumerate(heatmaps):
                        comet_logger.log_image(heatmap, name=uid+"L"+str(l_idx), step=time_step)
            #Log other metrics.
            else:
                if not np.isnan(value): 
                    comet_logger.log_metric(key, value, time_step)

    