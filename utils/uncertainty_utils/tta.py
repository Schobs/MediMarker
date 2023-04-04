import random
import numpy as np
import torch
import math
from scipy import ndimage
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt

def inverse_heatmaps(logged_vars, output, data_dict):
    """
    Wrapper function to invert a batch of heatmaps, depending on its transformation.
    """
    batch_size = len(data_dict['image'])
    for img_count in range(len(output)):
        try:
            transformation = logged_vars[img_count]['transform']
        except:
            transformation = logged_vars[img_count+batch_size]['transform']
        if "normal" in transformation:
            continue
        for heatmap_count in range(len(output[img_count])):
            heatmap = output[img_count][heatmap_count]
            org_heatmap = invert_heatmap(heatmap, transformation)
            output[img_count][heatmap_count] = org_heatmap
    return output

def invert_heatmap(heatmap, transformation):
    """
    Function to invert a heatmap from an applied transformation.
    """
    transform = list(transformation.keys())[0]
    if transform is "inverse_rotate":
        mag = transformation[transform] * -1
        temp = heatmap.cpu().numpy()
        new_heatmap = torch.from_numpy(ndimage.rotate(temp, mag, reshape=False))
        return new_heatmap
    elif transform is "inverse_flip":
        new_heatmap = torch.flipud(heatmap)
        return new_heatmap
    elif transform is "inverse_fliplr":
        new_heatmap = torch.fliplr(heatmap)
        return new_heatmap
    return heatmap

def invert_coordinates(orginal_coords, log_dict, img_size=[512, 512]):
    """
    A function to invert the TTA processes laid out in a previous function. Note that only the rotate function has an effect of transforming the coords therefore it is the
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
    for funct, coords_all, img_shape in zip(log_dict['individual_results'][-orginal_coords.shape[0]:], orginal_coords, img_size):
        if type(img_shape) == int:
            img_shape = [512, 512]
        key = list(funct['transform'].keys())[0]
        if "normal" in key:
            coords = coords_all
        elif "inverse_rotate" in key:
            coords = torch.stack([extract_original_coords_from_rotation(
                funct['transform']["inverse_rotate"], coords, img_shape) for coords in coords_all])
        elif "inverse_flip" in key:
            coords = torch.stack([extract_original_coords_from_flipud(coords, img_shape)
                                  for coords in coords_all])
        else:
            coords = torch.stack([extract_original_coords_from_fliplr(coords, img_shape)
                                  for coords in coords_all])
        inverted_predicted_coords.append(coords)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inverted_predicted_coords = torch.stack(inverted_predicted_coords).to(device)
    return inverted_predicted_coords

def extract_original_coords_from_rotation(rotation_mag, rotated_coords, training_resolution=[512, 512]):
    """
    Extracts the original coords from a rotated image - does assume the layout of eval_log to contain the coords in the eval_log["individual_results"]['predicted_coords'] section
    to be in a tuple i.e. (x1, y1).
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
    rotation_mag = rotation_mag * -1
    if type(rotated_coords) == torch.Tensor:
        rotated_coords = rotated_coords.cpu().numpy()
    x = rotated_coords[0]
    y = rotated_coords[1]
    try:
        x_scale = training_resolution[0][0]
        y_scale = training_resolution[1][0]
    except:
        x_scale = training_resolution[0]
        y_scale = training_resolution[1]
    centre_x = x_scale / 2
    centre_y = y_scale / 2
    theta_rad = np.deg2rad(rotation_mag)
    rot_mat = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
                        [np.sin(theta_rad), np.cos(theta_rad)]])
    p = np.array([x - centre_x, y - centre_y])
    p = rot_mat.dot(p)
    p = p + np.array([centre_x, centre_y])
    x_new = p[0]
    y_new = p[1]
    conv_coords = torch.tensor([x_new, y_new])
    return conv_coords

def extract_original_coords_from_flipud(flip_coords, training_resolution=[512, 512]):
    """"""
    if type(flip_coords) == torch.Tensor:
        flip_coords = flip_coords.cpu().numpy()
    original_x = flip_coords[0]
    original_y = training_resolution[1] - flip_coords[1]
    conv_coords = torch.tensor([original_x, original_y])
    return conv_coords

def extract_original_coords_from_fliplr(flip_coords, training_resolution=[512, 512]):
    """"""
    if type(flip_coords) == torch.Tensor:
        flip_coords = flip_coords.cpu().numpy()
    original_x = training_resolution[0] - flip_coords[0]
    original_y = flip_coords[1]
    conv_coords = torch.tensor([original_x, original_y])
    return conv_coords

def apply_tta_augmentation(data, seed):
    """
    A function that applies a random TTA transformation on an inputted image. Takes a random seed to determine the specific transformation and
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
    functs_list = ["flipud", "fliplr"] #rotate
    function_index = math.floor(seed / 100000 * 2)
    function_name = functs_list[function_index]
    img = data.cpu().detach().numpy()
    img_dims = img.shape
    img = np.reshape(img, (img_dims[1], img_dims[2], img_dims[0]))
    if function_name == "rotate":
        negative_sign = True if math.floor(seed / 2) == 0 else False
        rotate_magnitude = math.floor(seed / 100000 * 45) if function_name == "rotate" else 0
        rotate_magnitude = rotate_magnitude * -1 if negative_sign and rotate_magnitude != 0 else rotate_magnitude
        inverse_transform = {"inverse_rotate": rotate_magnitude}
        augemented_img = iaa.Rotate(rotate_magnitude).augment_image(img)
    elif function_name == "flipud":
        augemented_img = iaa.Flipud(1).augment_image(img)
        inverse_transform = {
            "inverse_flip": 1
        }
    elif function_name == "fliplr":
        augemented_img = iaa.Fliplr(1).augment_image(img)
        inverse_transform = {
            "inverse_fliplr": 1
        }
    else:
        return data, {"normal": None}
    print(inverse_transform)
    augemented_img = np.reshape(augemented_img, (img_dims[0], img_dims[1], img_dims[2]))
    data = torch.from_numpy(augemented_img.copy())
    return data, inverse_transform
