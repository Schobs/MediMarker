import numpy as np
import torch
from utils.im_utils.heatmap_manipulation import get_coords
from utils.uncertainty_utils.tta import extract_original_coords_from_flipud, extract_original_coords_from_fliplr, extract_coords_from_movevertical, extract_coords_from_movehorizontal


class EnsembleUncertainties():
    """ Class to calculate the predicted coordinates and uncertainty metrics for an ensemble of models. 
        The uncertainty metrics calculated are defined by the keys in the yaml config file (default: smha, ecpv, emha).

    """
    def __init__(self, uncertainty_keys, smha_choice_idx, landmarks):
        self.uncertainty_keys = uncertainty_keys
        self.smha_choice_idx = smha_choice_idx
        self.landmarks = landmarks


    def ensemble_inference_with_uncertainties(self, evaluation_logs, tta=False):
        """ Function to calculate the predicted coordinates and uncertainty metrics for an ensemble of models over a dictionary of evaluation logs.
        Args:
            evaluation_logs (dict): Dictionary of samples of the template from DictLogger.ensemble_inference_log_template():
        Returns:
            ensemble_results_dict (dict): Dictionary of the ensemble results with the following keys:
                - "smha": List of dicts per sample of the predicted coordinates and uncertainty metrics for the model with the highest maximum heatmap value.
                - "ecpv": List of dicts per sample of the predicted coordinates and uncertainty metrics using the variance of the ensembles.
                - "emha": List of dicts per sample of the predicted coordinates and uncertainty metrics using the maximum heatmap value of the average heatmap.
            ind_errors (dict): Dictionary of the individual landmark errors for each uncertainty metric.    

        """
        #Group ensemble predictions by uid
        sorted_dict_by_uid = {}
        if tta:
            for ind_res in evaluation_logs["individual_results"][1::2]: #NOT A STABLE SOLUTION; WILL HAVE TO DO FOR NOW - links to batch size in tta_inference set to 1
                uid = ind_res["uid"]
                if uid not in sorted_dict_by_uid:
                    sorted_dict_by_uid[uid] = {}
                    sorted_dict_by_uid[uid]["ind_preds"] = []
                sorted_dict_by_uid[uid]["ind_preds"].append(ind_res)
        else:
            for ind_res in evaluation_logs["individual_results"]:
                uid = ind_res["uid"]
                if uid not in sorted_dict_by_uid:
                    sorted_dict_by_uid[uid] = {}
                    sorted_dict_by_uid[uid]["ind_preds"] = []
                sorted_dict_by_uid[uid]["ind_preds"].append(ind_res)
        #Initialise results dictionaries
        all_ensemble_results_dicts = {uncert_key: [] for uncert_key in self.uncertainty_keys}
        ind_errors = {uncert_key: [[] for x in range(len(self.landmarks))] for uncert_key in self.uncertainty_keys}


        
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
                # target_coords_original_resolution = individual_results[0]["target_coords_original_resolution"]
            else:
                target_coords = None

            ################## 1) S-MHA ###########################
            if "smha" in self.uncertainty_keys:
                ensemble_results_dict, ind_errors = self.calculate_smha(individual_results, ensemble_results_dict, ind_errors, calc_error)

            ################## 2) E_CPV ###########################
            if "ecpv" in self.uncertainty_keys:
                if tta:
                    ensemble_results_dict, ind_errors = self.calculate_ecpv(individual_results, ensemble_results_dict, ind_errors, target_coords, calc_error, evaluation_logs["individual_results"][0::2])
                else:
                    ensemble_results_dict, ind_errors = self.calculate_ecpv(individual_results, ensemble_results_dict, ind_errors, target_coords, calc_error)
          
            ################## 3) E_MHA ###########################
            if "emha" in self.uncertainty_keys:
                ensemble_results_dict, ind_errors = self.calculate_emha(individual_results, ensemble_results_dict, ind_errors, target_coords, calc_error)

     

            #Add sample results to results dictionary.
            for k_ in list(all_ensemble_results_dicts.keys()):
                all_ensemble_results_dicts[k_].append(ensemble_results_dict[k_])

            # print("ensemble results: ", ensemble_results_dict)
        return all_ensemble_results_dicts, ind_errors

    def calculate_smha(self, individual_results, ensemble_results_dict, ind_errors, calc_error):
        """ Function to calculate the predicted coordinates and uncertainty metrics for the model with the highest maximum heatmap value.

        Args:
            individual_results (Dict): Dictionary of the individual landmark results of a sample, following the template of 
                DictLogger.log_key_variables() under the "individual_results" key, plus the extra keys from 
                DictLogger.ensemble_inference_log_template() under the "individual_results_extra_keys" key.
            ensemble_results_dict (Dict): Dictionary we are saving the uncertainty and coordinate info to per sample.
            ind_errors (Dict): Dictionary of the individual landmark errors for each uncertainty metric.
            calc_error (bool): Whether to calculate the error metrics or not (if the sample has annotation we do).

        Returns:
            ensemble_results_dict (Dict): Updated dictionary we are saving the uncertainty and coordinate info to per sample.
            ind_errors (Dict): Updated dictionary of the individual landmark errors for each uncertainty metric.
        """

        ensemble_results_dict["smha"]["predicted_coords"] = individual_results[self.smha_choice_idx]["predicted_coords"]
        ensemble_results_dict["smha"]["uncertainty"] = individual_results[self.smha_choice_idx]["hm_max"]
       
        #Save inference resolution results for each landmark individually
        for c_idx, pred_coord_t in enumerate(individual_results[self.smha_choice_idx]["predicted_coords"]):
            ensemble_results_dict["smha"]["predicted_L"+str(c_idx)] = pred_coord_t
            ensemble_results_dict["smha"]["uncertainty_L"+str(c_idx)] = ensemble_results_dict["smha"]["uncertainty"][c_idx][0]

        ####Calculate the error if annotation available
        if calc_error:
            #Error for inference resolution
            ensemble_results_dict["smha"]["ind_errors"] = individual_results[self.smha_choice_idx]["ind_errors"]
            ensemble_results_dict["smha"]["Error All Mean"] = np.mean(individual_results[self.smha_choice_idx]["ind_errors"])
            ensemble_results_dict["smha"]["Error All Std"] = np.std(individual_results[self.smha_choice_idx]["ind_errors"])

            for cidx, p_coord in enumerate(individual_results[self.smha_choice_idx]["ind_errors"]):
                ensemble_results_dict["smha"]["L"+str(cidx)] = p_coord
                ensemble_results_dict["smha"]["uncertainty L"+str(cidx)] = individual_results[self.smha_choice_idx]["hm_max"][cidx]
                ind_errors["smha"][cidx].append(p_coord)

        return ensemble_results_dict, ind_errors

    def calculate_ecpv(self, individual_results, ensemble_results_dict, ind_errors, target_coords, calc_error, tta=None):
        """ Function to calculate the predicted coordinates and uncertainty metrics using the variance between models.

        Args:
            individual_results (Dict): Dictionary of the individual landmark results of a sample, following the template of 
            DictLogger.log_key_variables() under the "individual_results" key, plus the extra keys from 
            DictLogger.ensemble_inference_log_template() under the "individual_results_extra_keys" key.
            ensemble_results_dict (Dict): Dictionary we are saving the uncertainty and coordinate info to per sample.
            ind_errors (Dict): Dictionary of the individual landmark errors for each uncertainty metric.
            target_coords (np.array): The target coordinates of the sample.
            calc_error (bool): Whether to calculate the error metrics or not (if the sample has annotation we do).

        Returns:
            ensemble_results_dict (Dict): Updated dictionary we are saving the uncertainty and coordinate info to per sample.
            ind_errors (Dict): Updated dictionary of the individual landmark errors for each uncertainty metric.
        """
        if tta is not None:
            for transforms, idx in zip(tta, [x for x in range(len(tta))]):
                if idx not in [2, 4]:
                    continue
                inverted_predicted_coords = []
                transform = transforms["transform"]
                img_shape = individual_results[idx]['original_image_size']
                coords_all = individual_results[idx]['predicted_coords']
                key = list(transforms['transform'].keys())[0]
                if "normal" in key:
                    continue
                elif "inverse_flip" in key:
                    coords = torch.stack([extract_original_coords_from_flipud(coords, img_shape)
                                        for coords in coords_all])
                elif "inverse_fliplr" in key:
                    coords = torch.stack([extract_original_coords_from_fliplr(coords, img_shape)
                                        for coords in coords_all])
                elif "inverse_movevertical" in key:
                    coords = torch.stack([extract_coords_from_movevertical(
                        transform["inverse_movevertical"], coords, img_shape) for coords in coords_all])
                elif "inverse_movehorizontal" in key:
                    coords = torch.stack([extract_coords_from_movehorizontal(
                        transform["inverse_movehorizontal"], coords, img_shape) for coords in coords_all])
                inverted_predicted_coords.append(coords)
                individual_results[idx]['predicted_coords'] = inverted_predicted_coords
            individual_results[2]['predicted_coords'] = individual_results[2]['predicted_coords'][0].detach().cpu().numpy()
            individual_results[4]['predicted_coords'] = individual_results[4]['predicted_coords'][0].detach().cpu().numpy()
        #Calculate the E-CPV
        average_coords = np.mean([dic['predicted_coords'] for dic in individual_results] , axis=0)
        # average_coords_og_resolution = np.mean([dic['predicted_coords_original_resolution'] for dic in individual_results] , axis=0)

        #the actual e-cpv uncertainty measure should be the same for both inference and resized to maintain the same scaling between all images.
        all_coord_vars = []
        for coord_idx, coord in enumerate(average_coords):
            all_coord_vars.append(np.mean([abs(np.linalg.norm(coord- x)) for x in [dic['predicted_coords'][coord_idx] for dic in individual_results]]))
                
        #import pdb; pdb.set_trace()
        ensemble_results_dict["ecpv"]["predicted_coords"] = np.round(average_coords)
        ensemble_results_dict["ecpv"]["uncertainty"] = all_coord_vars

        #Save inference resolution results for each landmark individually
        for c_idx, pred_coord_t in enumerate(ensemble_results_dict["ecpv"]["predicted_coords"]):
            ensemble_results_dict["ecpv"]["predicted_L"+str(c_idx)] = pred_coord_t
            ensemble_results_dict["ecpv"]["uncertainty_L"+str(c_idx)] = ensemble_results_dict["ecpv"]["uncertainty"][c_idx]


        ####Calculate the error if annotation available
        if calc_error:
            #Error for inference resolution
            ecpv_errors = np.linalg.norm(ensemble_results_dict["ecpv"]["predicted_coords"] - target_coords, axis=1)

            ensemble_results_dict["ecpv"]["ind_errors"] =ecpv_errors
            ensemble_results_dict["ecpv"]["Error All Mean"] = np.mean(ecpv_errors)
            ensemble_results_dict["ecpv"]["Error All Std"] = np.std(ecpv_errors)

            for cidx, p_coord in enumerate(ecpv_errors):
                ensemble_results_dict["ecpv"]["L"+str(cidx)] = p_coord
                ensemble_results_dict["ecpv"]["uncertainty L"+str(cidx)] = all_coord_vars[cidx]
                ind_errors["ecpv"][cidx].append(p_coord)
        return ensemble_results_dict, ind_errors


    def calculate_emha(self, individual_results, ensemble_results_dict, ind_errors, target_coords, calc_error):
        """ Function to calculate the predicted coordinates and uncertainty metrics using the average heatmap.

        Args:
            individual_results (Dict): Dictionary of the individual landmark results of a sample, following the template of 
            DictLogger.log_key_variables() under the "individual_results" key, plus the extra keys from 
            DictLogger.ensemble_inference_log_template() under the "individual_results_extra_keys" key.
            ensemble_results_dict (Dict): Dictionary we are saving the uncertainty and coordinate info to per sample.
            ind_errors (Dict): Dictionary of the individual landmark errors for each uncertainty metric.
            target_coords (np.array): The target coordinates of the sample.
            calc_error (bool): Whether to calculate the error metrics or not (if the sample has annotation we do).

        Returns:
            ensemble_results_dict (Dict): Updated dictionary we are saving the uncertainty and coordinate info to per sample.
            ind_errors (Dict): Updated dictionary of the individual landmark errors for each uncertainty metric.
        """

        #import pdb; pdb.set_trace()
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
        
        pred_coords =  np.round(pred_coords).cpu().detach().numpy()[0]
        max_values = max_values.cpu().detach().numpy()[0]
        
        ensemble_results_dict["emha"]["predicted_coords"] = pred_coords
        ensemble_results_dict["emha"]["uncertainty"] = max_values

           
        #Save inference resolution results for each landmark individually
        for c_idx, pred_coord_t in enumerate(ensemble_results_dict["emha"]["predicted_coords"]):
            ensemble_results_dict["emha"]["predicted_L"+str(c_idx)] = pred_coord_t
            ensemble_results_dict["emha"]["uncertainty_L"+str(c_idx)] = ensemble_results_dict["emha"]["uncertainty"][c_idx][0]
    
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


        return ensemble_results_dict, ind_errors
