"""
Module containing unit tests for functions in the 'tta' utils module.

Each test function tests a specific function from the 'tta' module using various input and output cases.

Author: Ethan Jones
"""

import sys
import os

import numpy as np
import torch

# Add the parent directory of the module to path... not sure of best practice here...
module_path = os.path.abspath(os.path.join('..', '/utils/uncertainty_utils'))
sys.path.append(module_path)

from tta import extract_original_coords_from_flipud, extract_original_coords_from_fliplr, extract_coords_from_movevertical, extract_coords_from_movehorizontal

def test_extract_original_coords_from_flipud():
    # Test case 1: Test with tensor input and default training resolution
    flip_coords = torch.tensor([100, 200])
    expected_output = torch.tensor([100, 312])
    output = extract_original_coords_from_flipud(flip_coords)
    assert torch.allclose(output, expected_output)

    # Test case 2: Test with tensor input and different training resolution
    flip_coords = torch.tensor([300, 400])
    expected_output = torch.tensor([300, 312])
    output = extract_original_coords_from_flipud(flip_coords, training_resolution=[600, 600])
    assert torch.allclose(output, expected_output)

    # Test case 3: Test with tensor input and different training resolution
    flip_coords = torch.tensor([50, 100])
    expected_output = torch.tensor([50, 156])
    output = extract_original_coords_from_flipud(flip_coords, training_resolution=[200, 200])
    assert torch.allclose(output, expected_output)

def test_extract_original_coords_from_fliplr():
    # Test case 1: Test with tensor input and default training resolution
    flip_coords = torch.tensor([100, 200])
    expected_output = torch.tensor([412, 200])
    output = extract_original_coords_from_fliplr(flip_coords)
    assert torch.allclose(output, expected_output)

    # Test case 2: Test with tensor input and different training resolution
    flip_coords = torch.tensor([300, 400])
    expected_output = torch.tensor([212, 400])
    output = extract_original_coords_from_fliplr(flip_coords, training_resolution=[600, 512])
    assert torch.allclose(output, expected_output)

    # Test case 3: Test with tensor input and different training resolution
    flip_coords = torch.tensor([50, 100])
    expected_output = torch.tensor([462, 100])
    output = extract_original_coords_from_fliplr(flip_coords, training_resolution=[1024, 512])
    assert torch.allclose(output, expected_output)

def test_extract_coords_from_movevertical():
    # Test case 1: Test with tensor input and default magnitude
    coords = torch.tensor([50, 100])
    magnitude = 0
    expected_output = torch.tensor([50, 100])
    output = extract_coords_from_movevertical(magnitude, coords)
    assert torch.allclose(output, expected_output)

    # Test case 2: Test with tensor input and different magnitude
    coords = torch.tensor([300, 200])
    magnitude = -50
    expected_output = torch.tensor([300, 150])
    output = extract_coords_from_movevertical(magnitude, coords)
    assert torch.allclose(output, expected_output)

    # Test case 3: Test with tensor input and different magnitude
    coords = torch.tensor([150, 400])
    magnitude = 100
    expected_output = torch.tensor([150, 500])
    output = extract_coords_from_movevertical(magnitude, coords)
    assert torch.allclose(output, expected_output)

def test_extract_coords_from_movehorizontal():
    # Test case 1: Test with tensor input and default magnitude
    coords = torch.tensor([50, 100])
    magnitude = 0
    expected_output = torch.tensor([50, 100])
    output = extract_coords_from_movehorizontal(magnitude, coords)
    assert torch.allclose(output, expected_output)

    # Test case 2: Test with tensor input and different magnitude
    coords = torch.tensor([300, 200])
    magnitude = -50
    expected_output = torch.tensor([250, 200])
    output = extract_coords_from_movehorizontal(magnitude, coords)
    assert torch.allclose(output, expected_output)
