"""
Utility module containing methods for the TTA implementation for uncertinty estimation

Author: Ethan Jones
Date: 2023-04-25
"""

import random
from typing import Union
import math

import numpy as np
import torch
from imgaug import augmenters as iaa

def inverse_heatmaps(logged_vars: dict, output: dict, data_dict: dict) -> dict:
    """
    Wrapper function to invert a batch of heatmaps, depending on its transformation.
    
    Parameters
    ---------
    `logged_vars` : Dict 
        Dictionary with logged transformation information for each image.
    `output`: Dict
        Batch of heatmaps to be inverted.
    `data_dict` : Dict
        Dictionary with data related to the batch of heatmaps, including the image data.
    
    Returns
    -------
    `output` : Dict
        Batch of inverted heatmaps.
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

def invert_heatmap(heatmap : Union[torch.Tensor, np.ndarray], transformation: dict) -> torch.Tensor:
    """
    Function to invert a heatmap from an applied transformation.

    Parameters
    --------
    `heatmap` : Union[torch.Tensor, np.ndarray]
        Heatmap to be inverted.
    `transformation` : Dict
        Dictionary with information about the applied transformation.
    
    Returns
    -------
    `heatmap` : torch.Tensor
        Inverted heatmap.
    """
    transform = list(transformation.keys())[0]
    if transform is "inverse_rotate":
        try:
            temp = heatmap.cpu().numpy()
        except:
            temp = heatmap
        mag = transformation[transform]
        new_heatmap = torch.from_numpy(iaa.Rotate(mag).augment_image(temp))
        return new_heatmap
    elif transform is "inverse_flip":
        new_heatmap = torch.flipud(heatmap)
        return new_heatmap
    elif transform is "inverse_fliplr":
        new_heatmap = torch.fliplr(heatmap)
        return new_heatmap
    elif transform is "inverse_movevertical":
        try:
            temp = heatmap.cpu().numpy()
        except:
            temp = heatmap
        mag = transformation[transform]
        new_heatmap = torch.from_numpy(iaa.TranslateY(px=(mag)).augment_image(temp))
    elif transform is "inverse_movehorizontal":
        try:
            temp = heatmap.cpu().numpy()
        except:
            temp = heatmap
        mag = transformation[transform]
        new_heatmap = torch.from_numpy(iaa.TranslateX(px=(mag)).augment_image(temp))
    return heatmap

def invert_coordinates(orginal_coords : list, log_dict : dict, img_size : list = [512, 512]) -> list:
    """
    A function to invert the TTA processes laid out in a previous function. 
    
    Note that only the rotate function has an effect of transforming the coords therefore it is the
    only function that needs inversing.

    Parameters
    ----------
    `orginal_coords` : list
        List of coordinates.
    `log_dict` : Dict
        Dictionary containing the results and transformation information.
    `img_size` : list (defaults to [512,512])
        Size of the image.
    
    Returns
    -------
    `inverted_predicted_coords` : list
        List of inverted coords.
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
        elif "inverse_fliplr" in key:
            coords = torch.stack([extract_original_coords_from_fliplr(coords, img_shape)
                                  for coords in coords_all])
        elif "inverse_movevertical" in key:
            coords = torch.stack([extract_coords_from_movevertical(
                funct['transform']["inverse_movevertical"], coords, img_shape) for coords in coords_all])
        elif "inverse_movehorizontal" in key:
            coords = torch.stack([extract_coords_from_movehorizontal(
                funct['transform']["inverse_movehorizontal"], coords, img_shape) for coords in coords_all])
        inverted_predicted_coords.append(coords)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inverted_predicted_coords = torch.stack(inverted_predicted_coords).to(device)
    return inverted_predicted_coords

def extract_original_coords_from_rotation(rotation_mag : float, rotated_coords : tuple, training_resolution : list = [512, 512]) -> torch.Tensor:
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
    `conv_coords` : torch.Tensor
        Again assumes the layout of the evaluation log as mentioned above. A tuple containing the converted coordinates now relating to the
        original image.
    """
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
    """
    Extracts the original coordinates of a flipped image based on the flipped coordinates and the training resolution.

    :param flip_coords: A tensor or numpy array representing the flipped coordinates of an image, with shape (2,).
    :type flip_coords: torch.Tensor or numpy.ndarray
    :param training_resolution: The resolution of the training images, represented as a list of width and height. Default is [512, 512].
    :type training_resolution: list
    :return: A tensor representing the original coordinates of the flipped image, with shape (2,).
    :rtype: torch.Tensor
    """
    if type(flip_coords) == torch.Tensor:
        flip_coords = flip_coords.cpu().numpy()
    original_x = flip_coords[0]
    original_y = training_resolution[1] - flip_coords[1]
    conv_coords = torch.tensor([original_x, original_y])
    return conv_coords


def extract_original_coords_from_fliplr(flip_coords, training_resolution=[512, 512]):
    """
    Extracts the original coordinates of a flipped image based on the flipped coordinates and the training resolution.

    :param flip_coords: A tensor or numpy array representing the flipped coordinates of an image, with shape (2,).
    :type flip_coords: torch.Tensor or numpy.ndarray
    :param training_resolution: The resolution of the training images, represented as a list of width and height. Default is [512, 512].
    :type training_resolution: list
    :return: A tensor representing the original coordinates of the flipped image, with shape (2,).
    :rtype: torch.Tensor
    """
    if type(flip_coords) == torch.Tensor:
        flip_coords = flip_coords.cpu().numpy()
    original_x = training_resolution[0] - flip_coords[0]
    original_y = flip_coords[1]
    conv_coords = torch.tensor([original_x, original_y])
    return conv_coords


def extract_coords_from_movevertical(magnitude, coords):
    """
    Extracts the coordinates of an image moved vertically based on the input magnitude and original coordinates.

    :param magnitude: The magnitude of the vertical movement.
    :type magnitude: float
    :param coords: A tensor or numpy array representing the original coordinates of an image, with shape (2,).
    :type coords: torch.Tensor or numpy.ndarray
    :param training_resolution: The resolution of the training images, represented as a list of width and height. Default is [512, 512].
    :type training_resolution: list
    :return: A tensor representing the coordinates of the moved image, with shape (2,).
    :rtype: torch.Tensor
    """
    if type(coords) == torch.Tensor:
        coords = coords.cpu().numpy()
    original_x = coords[0]
    original_y = coords[1] + magnitude
    conv_coords = torch.tensor([original_x, original_y])
    return conv_coords


def extract_coords_from_movehorizontal(magnitude, coords):
    """
    Extracts new coordinates by moving the input coordinates horizontally by a certain magnitude.

    Parameters:
        magnitude (int or float): The amount to move the coordinates horizontally.
        coords (torch.Tensor or numpy.ndarray): The original coordinates to move.

    Returns:
        torch.Tensor: The new coordinates after moving horizontally.
    """
    if type(coords) == torch.Tensor:
        coords = coords.cpu().numpy()
    original_x = coords[0] + magnitude
    original_y = coords[1]
    conv_coords = torch.tensor([original_x, original_y])
    return conv_coords

def apply_tta_augmentation(data, seed):
    """
    A function that applies a random TTA transformation on an inputted image. Takes a random seed to determine the specific transformation and
    the magnitude of such transformation using a basic mapping function with the boundaries provided by the imgaug documentation for each
    specific transformation (hence the difference in allowed values). Also determines randomly whether the allowed transformations should be
    multiplied by -1 to allow for wider range of possible transforms.

    Note: Rotation is depricated.

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
    functs_list = ["flipud", "fliplr", "movevertical", "movehorizontal"]
    function_index = math.floor(seed / 100000 * len(functs_list))
    function_index = 0
    function_name = functs_list[function_index]
    img = data.cpu().detach().numpy()
    img_dims = img.shape
    img = np.reshape(img, (img_dims[1], img_dims[2], img_dims[0]))
    if function_name == "rotate":
        max_rotation = 5
        negative_sign = True if math.floor(seed / 2) == 0 else False
        rotate_magnitude = math.floor(seed / 100000 * max_rotation)
        if rotate_magnitude == 0:
            rotate_magnitude = random.randint(1, max_rotation)
        rotate_magnitude = rotate_magnitude * -1 if negative_sign and rotate_magnitude != 0 else rotate_magnitude
        inverse_transform = {"inverse_rotate": rotate_magnitude * -1}
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
    elif function_name == "movevertical":
        max_translation = 5
        move_magnitude = math.floor(seed / 100000 * max_translation)
        if move_magnitude == 0:
            move_magnitude = random.randint(1, max_translation)
        negative_sign = True if math.floor(seed / 2) == 0 else False
        move_magnitude = move_magnitude * -1 if negative_sign and move_magnitude != 0 else move_magnitude
        augemented_img = iaa.TranslateY(px=(move_magnitude)).augment_image(img)
        inverse_transform = {
            "inverse_movevertical": move_magnitude * -1
        }
    elif function_name == "movehorizontal":
        max_translation = 5
        move_magnitude = math.floor(seed / 100000 * max_translation)
        if move_magnitude == 0:
            move_magnitude = random.randint(1, max_translation)
        negative_sign = True if math.floor(seed / 2) == 0 else False
        move_magnitude = move_magnitude * -1 if negative_sign and move_magnitude != 0 else move_magnitude
        augemented_img = iaa.TranslateX(px=(move_magnitude)).augment_image(img)
        inverse_transform = {
            "inverse_movehorizontal": move_magnitude * -1
        }
    else:
        return data, {"normal": None}
    augemented_img = np.reshape(augemented_img, (img_dims[0], img_dims[1], img_dims[2]))
    data = torch.from_numpy(augemented_img.copy())
    return data, inverse_transform
