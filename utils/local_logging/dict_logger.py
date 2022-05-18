import torch
import numpy as np
import copy

class DictLogger():
    """ A dictionary based logger to save results. Extend this class to log any extra variables!
    """
    def __init__(self, num_landmarks, is_regressing_sigma, multi_part_loss_keys):
        #Device

        self.num_landmarks = num_landmarks
        self.is_regressing_sigma = is_regressing_sigma
        self.multi_part_loss_keys = multi_part_loss_keys
        
        self.per_epoch_logs = self.per_epoch_log_template()
        self.evaluation_logged_vars = self.evaluation_log_template()



    def per_epoch_log_template(self):
        logged_per_epoch =  {"valid_coord_error_mean": [], "epoch_time": []}
        if self.is_regressing_sigma:
            logged_per_epoch["sigmas_mean"] =  []
            for sig in range(self.num_landmarks):
                 logged_per_epoch["sigma_"+str(sig)] = []

        #initialise keys for logging the multi-part losses.
        for key_ in self.multi_part_loss_keys:
            logged_per_epoch["training_" + key_] = []
            logged_per_epoch["validation_" + key_] = []

        return logged_per_epoch

    def evaluation_log_template(self):
        return {"individual_results": [], "landmark_errors": [[] for x in range(self.num_landmarks)]}

    def get_epoch_logger(self):
        return copy.deepcopy(self.per_epoch_logs)

    def get_evaluation_logger(self):
        return copy.deepcopy(self.evaluation_logged_vars)

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
                
            #2) If log_coords, get coords from output. Then, check for the keys for what to log.
            if log_coords:
                
                coord_error = torch.linalg.norm((pred_coords- target_coords), axis=2)
                

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
                        ind_dict["Error All Mean"] = (np.mean(coord_error[idx].detach().cpu().numpy()))
                        ind_dict["Error All Std"] = (np.std(coord_error[idx].detach().cpu().numpy()))
                        ind_dict["ind_errors"] = ((coord_error[idx].detach().cpu().numpy()))
                        ind_dict["predicted_coords"] = ((pred_coords[idx].detach().cpu().numpy()))
                        ind_dict["target_coords"] = ((target_coords[idx].detach().cpu().numpy()))
                        ind_dict["uid"] = ((data_dict["uid"][idx]))

                        for coord_idx, er in enumerate(coord_error[idx]):
                            ind_dict["L"+str(coord_idx)] = er.detach().cpu().numpy()

                            if "landmark_errors" in vars_to_log:
                                log_dict["landmark_errors"][coord_idx].append(er.detach().cpu().numpy())

                        #any extra info returned by the child class when calculating coords from outputs e.g. heatmap_max
                        for key_ in list(extra_info.keys()):
                         

                            ind_dict[key_] = ((extra_info[key_][idx].detach().cpu().numpy()))


                        log_dict["individual_results"].append(ind_dict)
            
            return log_dict

    def log_epoch_end_variables(self, per_epoch_logs, time, sigmas ):
        """Logs end of epoch variables. If given a list of things to log it generates the mean of the list.

        Args:
            per_epoch_logs (Dict): Dict of variables to log.
            time (float): time it took for epoch
            sigmas ([Tensor]): Sigmas for the heatmap.
        """

        if "epoch_time" in list(per_epoch_logs.keys()):
            per_epoch_logs["epoch_time"] =  time
        if "sigmas_mean" in list(per_epoch_logs.keys()):
            np_sigmas = [x.cpu().detach().numpy() for x in sigmas]
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
            
        return per_epoch_logs
    

    def log_dict_to_comet(self, comet_logger, dict_to_log, time_step):
        for key, value in dict_to_log.items():
            comet_logger.log_metric(key, value, time_step)
    