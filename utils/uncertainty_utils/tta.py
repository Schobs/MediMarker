import numpy as np
import torch
import math


def invert_coordinates(orginal_coords, log_dict):
    """
    A function to invert the TTA processes laid out in a previous function. Note that only teh rotate function has an effect of transforming the coords therefore it is the
    only function that needs inversing.
    Parameters
    ----------
    `eval_logs` : List of Dicts
        The evaluation logs of layout: {"individual_results": [], "landmark_errors": [[] for x in range(self.num_landmarks)],
                                        "landmark_errors_original_resolution": [[] for x in range(self.num_landmarks)],
                                        "sample_info_log_keys": self.standard_info_keys, "individual_results_extra_keys": ['hm_max', 'coords_og_size']}.
    `inverse_functs` : Dict
        A dict containing the names and magnitudes of each applied TTA function.
    Returns
    -------
    `eval_logs` : List of Dicts
        The now altered evaluation logs with the updated, and inversed, coordinates from the TTA processes.
    """
    inverted_predicted_coords = []
    for funct, coords_all in zip(log_dict['individual_results'][-orginal_coords.shape[0]:], orginal_coords):
        key = funct['transform'].keys()
        if "normal" in key:
            coords = coords_all
        elif "inverse_rotate" in key:
            coords = torch.stack([extract_original_coords_from_rotation(
                funct['transform']["inverse_rotate"], coords) for coords in coords_all])
        elif "inverse_flip" in key:
            coords = torch.stack([extract_original_coords_from_flipud(coords)
                                  for coords in coords_all])
        elif "inverse_scalex" in key:
            coords = torch.stack([extract_original_coords_from_scale_x(funct['transform']["inverse_scalex"], coords)
                                  for coords in coords_all])
        else:
            coords = torch.stack([extract_original_coords_from_scale_y(funct['transform']["inverse_scaley"], coords)
                                  for coords in coords_all])

        inverted_predicted_coords.append(coords)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inverted_predicted_coords = torch.stack(inverted_predicted_coords).to(device)
    return inverted_predicted_coords


def extract_original_coords_from_rotation(rotation_mag, rotated_coords, training_resolution=[512, 512]):
    """
    Extracts the original coords from a rotated image - does assume the layout of eval_log to contain the coords in the eval_log["individual_results"]['predicted_coords'] section
    to be in a tuple i.e. [(x1, y1)].
    References:
    - https://study.com/skill/learn/how-to-find-the-coordinates-of-a-polygon-after-a-rotation-explanation.html
    Parameters:
    -----------
    `rotation_mag` : Float
        The rotation magnitude of the transformation. Note that this will be inversed within this function.
    `rotated_coords` : Tuple
        The coords from the rotated image i.e. (x1, y1)
    Returns
    -------
    `conv_coords` : Tuple
        Again assumes the layout of the evaluation log as mentioned above. A tuple containing the converted coordinates now relating to the
        original image.
    """
    deg = math.radians(-1 * rotation_mag)
    x = rotated_coords[0]
    y = rotated_coords[1]
    xo = training_resolution[0] / 2
    yo = training_resolution[1] / 2
    x_final = math.cos(deg) * (x - xo) - math.sin(deg) * (y - yo) + xo
    y_final = math.sin(deg) * (x - xo) + math.cos(deg) * (y - yo) + yo
    conv_coords = torch.tensor([x_final, y_final])
    return conv_coords


def extract_original_coords_from_scale_x(scale_mag, scaled_coords, training_resolution=[512, 512]):
    """"""
    original_x = scaled_coords[0] / scale_mag
    original_y = scaled_coords[1]
    conv_coords = torch.tensor([original_x, original_y])
    return conv_coords


def extract_original_coords_from_scale_y(scale_mag, scaled_coords, training_resolution=[512, 512]):
    """"""
    original_x = scaled_coords[0]
    original_y = scaled_coords[1] / scale_mag
    conv_coords = torch.tensor([original_x, original_y])
    return conv_coords


def extract_original_coords_from_flipud(flip_coords, training_resolution=[512, 512]):
    """"""
    original_x = flip_coords[0]
    original_y = training_resolution[1] - flip_coords[1]
    conv_coords = torch.tensor([original_x, original_y])
    #import pdb; pdb.set_trace()
    return conv_coords
