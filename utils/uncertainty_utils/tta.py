import random
import numpy as np
import torch
import math
from imgaug import augmenters as iaa


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


def apply_tta_augmentation(data, seed, idx):
    """
    A function that applies a random TTA trnasformation on an inputted image. Takes a random seed to determine the specific transformation and
    the magnitude of such transformation using a basic mapping function with the boundaries provided by the imgaug documentation for each
    specific transformation (hence the difference in allowed values). Also determines randomly whether the allowed transformations should be
    multiplied by -1 to allow for wider range of possible transforms.
    Parameters
    ----------
    `data` : List
        The dict containing the features and information of the image in question.
    `seed` : Int
        A stochastic seed variable used as described as above.
    Returns
    -------
    `augemented_dict` : Dict
        The augmented image.
    `inverse_transform` : Dict
        With the key being the name of the augmentation transform, and the value being the magnitude with which the transform was applied.
    """
    functs_list = ["rotate", "scalex", "scaley", "flipud"]
    function_index = math.floor(seed / 100000 * 4)
    function_name = functs_list[function_index]
    negative_sign = True if seed / 2 == 0 else False
    rotate_magnitude = math.floor(seed / 10000 * 360) if function_name == "rotate" else 0
    scale_magnitude = round(random.uniform(0.8, 1.2), 2) if function_name == "scalex" or "scaley" else 0
    rotate_magnitude = rotate_magnitude * -1 if negative_sign and rotate_magnitude != 0 else rotate_magnitude
    functions = {
        "rotate": iaa.Sequential([iaa.Rotate(rotate_magnitude)]),
        "scalex": iaa.Sequential([iaa.ScaleX(scale_magnitude)]),
        "scaley": iaa.Sequential([iaa.ScaleY(scale_magnitude)]),
        "flipud": iaa.Sequential([iaa.Flipud(1)])
    }
    inverse_functions = {
        "inverse_rotate": rotate_magnitude,
        "inverse_scalex": scale_magnitude,
        "inverse_scaley": scale_magnitude,
        "inverse_flip": 1
    }
    transform = functions[function_name]
    inverse_transform = {list(inverse_functions.keys())[function_index]:
                         list(inverse_functions.values())[function_index]}
    img = data['image'][idx].cpu().detach().numpy()
    img_dims = img.shape
    img = np.reshape(img, (img_dims[1], img_dims[2], img_dims[0]))
    augemented_img = transform.augment_image(img)
    augemented_img = np.reshape(augemented_img, (img_dims[0], img_dims[1], img_dims[2]))
    data['image'][idx] = torch.from_numpy(augemented_img.copy())
    return data, inverse_transform
