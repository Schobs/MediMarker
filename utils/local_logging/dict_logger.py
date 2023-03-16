import torch
import numpy as np
import copy

from utils.uncertainty_utils.tta import invert_coordinates


class DictLogger():
    """ A dictionary based logger to save results. Extend this class to log any extra variables!
    """

    def __init__(self, num_landmarks, is_regressing_sigma, multi_part_loss_keys, additional_sample_attribute_keys):
        # Device

        self.num_landmarks = num_landmarks
        self.is_regressing_sigma = is_regressing_sigma
        self.multi_part_loss_keys = multi_part_loss_keys
        self.add_sample_att_keys = additional_sample_attribute_keys
        self.standard_info_keys = ["uid", "full_res_coords", "annotation_available",
            "image_path", "target_coords",  "resizing_factor", "original_image_size"]

        self.per_epoch_logs = self.per_epoch_log_template()
        self.evaluation_logged_vars = self.evaluation_log_template()
        self.ensemble_inference_logs = self.ensemble_inference_log_template()
        self.tta_inference_logs = self.tta_inference_log_template()

        # also add the additional sample attributes to the standard info keys.
        self.standard_info_keys = self.standard_info_keys + self.add_sample_att_keys

    def per_epoch_log_template(self):
        logged_per_epoch = {"valid_coord_error_mean": [], "epoch_time": [],
            "lr": [], "individual_results_extra_keys": []}
        if self.is_regressing_sigma:
            logged_per_epoch["sigmas_mean"] = []
            for sig in range(self.num_landmarks):
                 logged_per_epoch["sigma_"+str(sig)] = []

        # initialise keys for logging the multi-part losses.
        for key_ in self.multi_part_loss_keys:
            logged_per_epoch["training_" + key_] = []
            logged_per_epoch["validation_" + key_] = []

        return logged_per_epoch

    def evaluation_log_template(self):
        return {"individual_results": [], "landmark_errors": [[] for x in range(self.num_landmarks)],
            "landmark_errors_original_resolution": [[] for x in range(self.num_landmarks)],
            "sample_info_log_keys": self.standard_info_keys, "individual_results_extra_keys": ['hm_max', 'coords_og_size']}

    def ensemble_inference_log_template(self):

        # print("standard_info_keys", standard_info_keys)
        return {"individual_results": [], "landmark_errors": [[] for x in range(self.num_landmarks)],
        "landmark_errors_original_resolution": [[] for x in range(self.num_landmarks)],
        "sample_info_log_keys": self.standard_info_keys, "individual_results_extra_keys": ['final_heatmaps', 'hm_max', 'coords_og_size']}

    def tta_inference_log_template(self):
        return {"individual_results": [], "landmark_errors": [[] for x in range(self.num_landmarks)],
        "landmark_errors_original_resolution": [[] for x in range(self.num_landmarks)],
        "sample_info_log_keys": self.standard_info_keys, "individual_results_extra_keys": ['tta_augmentations', 'final_heatmaps', 'hm_max', 'coords_og_size']}

    def get_epoch_logger(self):
        return copy.deepcopy(self.per_epoch_logs)

    def get_evaluation_logger(self):
        return copy.deepcopy(self.evaluation_logged_vars)

    def get_ensemble_inference_logger(self):
        return copy.deepcopy(self.ensemble_inference_logger)

    def log_key_variables(self, log_dict, pred_coords, extra_info, target_coords, loss_dict, data_dict, log_coords, split):
        """Logs base key variables. Should be extended by child class for non-generic logging variables.

        Args:
            output (_type_): _description_
            loss (_type_): _description_
            data_dict (_type_): _description_
            logged_vars (_type_): _description_
        """
        # 1) Log training/validation losses based on split.
        vars_to_log = list(log_dict.keys())
        for key, value in loss_dict.items():
            key_ = split + "_" + key

            if key_ in vars_to_log:
                log_dict[key_].append(value)

        # Only log info we requested in the evaluation/ensemble templates
        # print("before extra info filter", extra_info.keys())
        extra_info = {k: extra_info[k] for k in log_dict['individual_results_extra_keys'] if k in extra_info}
        # print("after extra info filter", extra_info.keys())

        if log_coords:

            if "individual_results_extra_keys" in vars_to_log and "tta_augmentations" in log_dict["individual_results_extra_keys"]:
                pred_coords = invert_coordinates(pred_coords, log_dict)
                # log_dict["individual_results"] = log_dict["individual_results"][:len(log_dict["individual_results"])-pred_coords.shape[0]]

            # Get coord error of the input resolution to network
            coord_error = torch.linalg.norm((pred_coords - target_coords), axis=2)

            if split == "validation":
                if "valid_coord_error_mean" in vars_to_log:
                    log_dict["valid_coord_error_mean"].append(np.mean(coord_error.detach().cpu().numpy()))
            elif split == "training":
                if "train_coord_error_mean" in vars_to_log:
                    log_dict["train_coord_error_mean"].append(np.mean(coord_error.detach().cpu().numpy()))

            # Save data for ecah sample individually
            if "individual_results" in vars_to_log:

                for idx in range(len(pred_coords)):
                    # @ETHAN so idx goes through each sample in the batch. If we have tta_augmentations,
                    # we are passing the augmentation into the log_dict["individual_results"] (which is kinda bad ),
                    # so we need to pop it out of the list to get the value and remove it.
                    # the -(len(pred_coords) means it will only ever get the tta_augmentations out
                    # and not the running list of results.
                    if "tta_augmentations" in log_dict["individual_results_extra_keys"]:
                        tta_aug = log_dict["individual_results"].pop(-(len(pred_coords)))
                        ind_dict = {"tta_aug": tta_aug}
                    else:
                        ind_dict = {}

                    # First log standard info about the sample, maybe detaching if it is a tensor
                    for standard_info_key in self.standard_info_keys:

                        data_point = data_dict[standard_info_key][idx]

                        if torch.is_tensor(data_point):
                            data_point = data_point.detach().cpu().numpy()

                        ind_dict[standard_info_key] = data_point


                    ind_dict["predicted_coords"] = ((pred_coords[idx].detach().cpu().numpy()))

                

                    # If target annotation not avaliable, we don't know the error
                    if ind_dict["annotation_available"] == False:

                        # Save for network input resolution
                        ind_dict["Error All Mean"] = None
                        ind_dict["Error All Std"] = None
                        ind_dict["ind_errors"] = None
                        ind_dict["target_coords"] = None

                        for coord_idx, er in enumerate(coord_error[idx]):
                            ind_dict["L"+str(coord_idx)] = None



                            
                    else:
                        # Save for network input resolution
                        ind_dict["Error All Mean"] = (np.mean(coord_error[idx].detach().cpu().numpy()))
                        ind_dict["Error All Std"] = (np.std(coord_error[idx].detach().cpu().numpy()))
                        ind_dict["ind_errors"] = ((coord_error[idx].detach().cpu().numpy()))
                        ind_dict["target_coords"] = ((target_coords[idx].detach().cpu().numpy()))


                        for coord_idx, er in enumerate(coord_error[idx]):
                            ind_dict["L"+str(coord_idx)] = er.detach().cpu().numpy()

                            if "landmark_errors" in vars_to_log:
                                log_dict["landmark_errors"][coord_idx].append(er.detach().cpu().numpy())

                    

                    # any extra info returned by the child class when calculating coords from outputs e.g. heatmap_max
                    for key_ in list(extra_info.keys()):
                        # print(extra_info[key_][idx])
                        # print("extra_info key_ ", key_, len(extra_info[key_]))
                        if "debug" not in key_:
                            # if key_ == "coords_og_size":
                            #     continue
                            # if type(extra_info[key_][idx]) != list:
                            ind_dict[key_] = ((extra_info[key_][idx].detach().cpu().numpy()))
                            
                            # else:
                            #     ind_dict[key_] = [x.detach().cpu().numpy() for x in ((extra_info[key_][idx].detach().cpu().numpy()))]


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
            # get the mean of all the batches from the training/validations. 
            if isinstance(value, list):
                per_epoch_logs[key] = np.round(np.mean([x.detach().cpu().numpy() if torch.is_tensor(x) else x for x in value]),5)
            if torch.is_tensor(value):
                per_epoch_logs[key] = np.round(value.detach().cpu().numpy(), 5)
            
        return per_epoch_logs
    

    def log_dict_to_comet(self, comet_logger, dict_to_log, time_step):
        for key, value in dict_to_log.items():
            comet_logger.log_metric(key, value, time_step)
    