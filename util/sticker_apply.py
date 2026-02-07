import copy

import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
from matplotlib import pyplot as plt
import os
from tqdm import tqdm
import pickle
from PIL import Image


def project(vehicle_image, filename, mask, project_point_dict):
    """
    :param vehicle_image: cv2 object
    :param filename: str
    :param mask: cv2 object
    :return ndarray [h, w, 3]
    """
    image_height, image_width = vehicle_image.shape[:2]

    points_3d_2d, flag = project_point_dict[filename]

    mask_resized = cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_LINEAR)

    # project mask
    h, w = mask_resized.shape[:2]
    points_2d = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    transform_matrix = cv2.getPerspectiveTransform(points_2d, points_3d_2d)

    transformed_mask_resized = cv2.warpPerspective(mask_resized, transform_matrix, (image_width, image_height),
                                                   flags=cv2.INTER_LINEAR)

    transformed_mask_resized_gray = cv2.cvtColor(transformed_mask_resized, cv2.COLOR_BGR2GRAY)
    _, transformed_mask_resized_binary = cv2.threshold(transformed_mask_resized_gray, 5, 255, cv2.THRESH_BINARY)

    # Apply the mask to the image
    masked_image = vehicle_image.copy()
    if not flag:
        masked_image[transformed_mask_resized_binary > 0] = transformed_mask_resized[
            transformed_mask_resized_binary > 0]

    return masked_image


def project_mask(vehicle_image, filename, mask, is_binary, project_point_dict):
    """
    :param vehicle_image: cv2 object
    :param filename: str
    :param mask: cv2 object
    :return ndarray [h, w, 3]
    """
    image_height, image_width = vehicle_image.shape[:2]

    points_3d_2d, flag = project_point_dict[filename]

    # if (points_3d_2d[-1] == points_3d_2d[-2]).all():
    #     print("Hit")

    mask_resized = cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_LINEAR)

    # project mask
    h, w = mask_resized.shape[:2]
    points_2d = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    transform_matrix = cv2.getPerspectiveTransform(points_2d, points_3d_2d)

    transformed_mask_resized = cv2.warpPerspective(mask_resized, transform_matrix, (image_width, image_height),
                                                   flags=cv2.INTER_LINEAR)
    if is_binary:
        transformed_mask_resized_gray = cv2.cvtColor(transformed_mask_resized, cv2.COLOR_BGR2GRAY)
        _, transformed_mask_resized_binary = cv2.threshold(transformed_mask_resized_gray, 5, 255, cv2.THRESH_BINARY)
        transformed_mask_resized = cv2.merge(
            [transformed_mask_resized_binary, transformed_mask_resized_binary, transformed_mask_resized_binary])

    return transformed_mask_resized, flag


def reverse_project_mask(transformed_mask, filename, original_size, project_point_dict):
    """
    Reverse project a transformed mask back to its original size.

    :param transformed_mask: cv2 object of the transformed mask.
    :param filename: str, filename to get pre-calculated points.
    :param original_size: tuple, original size of the mask (width, height).
    :return ndarray [h, w, 3], reversed mask.
    """
    # Retrieve pre-defined points from the dictionary
    points_3d_2d, _ = project_point_dict[filename]

    # Calculate the original 2D points of the mask
    h, w = original_size
    points_2d = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    # Inverse transform
    inverse_transform_matrix = cv2.getPerspectiveTransform(points_3d_2d, points_2d)
    reversed_mask = cv2.warpPerspective(transformed_mask, inverse_transform_matrix,
                                        (w, h), flags=cv2.INTER_LINEAR)

    return reversed_mask


def adjust_overlapping_points(points, adjustment=1):
    """
    Adjust points if they overlap.

    :param points: ndarray of shape (n, 2) representing n points.
    :param adjustment: The small distance to adjust overlapping points by.
    :return: Adjusted points.
    """
    n = points.shape[0]
    # adjusted_points = points.clone()

    for i in range(n):
        for j in range(i + 1, n):
            if (points[i] == points[j]).all():
                # print("Overlap, adjusted")
                points[j] += adjustment

    return points


def project_mask_pytorch(vehicle_image, filename, mask, is_binary, project_point_dict):
    """
    :param vehicle_image: PIL image or Tensor
    :param filename: str
    :param mask: PIL image or Tensor
    :param is_binary: bool
    :param project_point_dict: dict
    :return Tensor [C, H, W]
    """
    # Convert images to tensor if they are not already
    if not isinstance(vehicle_image, torch.Tensor):
        vehicle_image = TF.to_tensor(vehicle_image)
    if not isinstance(mask, torch.Tensor):
        mask = TF.to_tensor(mask)

    image_height, image_width = vehicle_image.shape[1:3]

    points_3d_2d, flag = project_point_dict[filename]

    # Resize mask
    mask_resized = TF.resize(mask, [image_height, image_width])

    # Define the points for perspective transformation
    h, w = mask_resized.shape[1:3]
    points_2d = [[0, 0], [w, 0], [w, h], [0, h]]

    # Perform perspective transform
    transformed_mask_resized = TF.perspective(mask_resized, startpoints=points_2d, endpoints=points_3d_2d)

    if is_binary:
        # Convert to grayscale and apply threshold
        transformed_mask_resized_gray = transformed_mask_resized.mean(dim=0, keepdim=True)
        transformed_mask_resized_binary = (transformed_mask_resized_gray > 0.02).float()

        # Merge channels
        transformed_mask_resized = transformed_mask_resized_binary.repeat(3, 1, 1)

    return transformed_mask_resized, flag


def resize_and_pad(tensor, img_size, return_size=False):
    batch_size, channels, height, width = tensor.shape
    max_size = max(height, width)
    ratio = img_size / max_size
    new_height = int(height * ratio)
    new_width = int(width * ratio)

    resized_tensor = F.interpolate(tensor, size=(new_height, new_width), mode='bilinear', align_corners=False)

    pad_top = (img_size - new_height) // 2
    pad_bottom = img_size - new_height - pad_top
    pad_left = (img_size - new_width) // 2
    pad_right = img_size - new_width - pad_left
    padded_tensor = F.pad(resized_tensor, (pad_left, pad_right, pad_top, pad_bottom), 'constant', value=0)

    assert padded_tensor.shape[-1] == padded_tensor.shape[-2] and padded_tensor.shape[-1] == img_size
    if return_size:
        return padded_tensor, new_height, new_width
    return padded_tensor


def resize_and_pad_tensors(tensor_tuple, img_size):
    # Process each tensor in the tuple independently
    processed_tensors = []
    for tensor in tensor_tuple:
        # Assuming tensor is of shape [channels, height, width]
        channels, height, width = tensor.shape
        max_size = max(height, width)
        ratio = img_size / max_size
        new_height = int(height * ratio)
        new_width = int(width * ratio)

        # Resize
        resized_tensor = F.interpolate(tensor.unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=False).squeeze(0)

        # Pad

        pad_top = (img_size - new_height) // 2
        pad_bottom = img_size - new_height - pad_top
        pad_left = (img_size - new_width) // 2
        pad_right = img_size - new_width - pad_left
        padded_tensor = F.pad(resized_tensor, (pad_left, pad_right, pad_top, pad_bottom), 'constant', value=128/255)

        processed_tensors.append(padded_tensor)

    # Stack all processed tensors into a single tensor
    stacked_tensor = torch.stack(processed_tensors, dim=0)

    return stacked_tensor


def project_mask_pytorch_batch(vehicle_images, filenames, mask, is_binary, project_point_dict, img_size):
    """
    :param vehicle_images: Tensor of shape [N, C, H, W]
    :param filenames: list of str
    :param mask: Tensor of shape [C, H, W]
    :param is_binary: bool
    :param project_point_dict: dict
    :param img_size: int, target size for padding
    :return Tuple(Tensor [N, C, img_size, img_size], List[flags])
    """
    transformed_masks = []
    flags = []

    for i in range(len(vehicle_images)):
        vehicle_image = vehicle_images[i]
        filename = filenames[i]

        # Get the size of the current vehicle image
        image_height, image_width = vehicle_image.shape[1:3]

        # Resize mask to match the size of the current vehicle image
        mask_resized = TF.resize(mask, [image_height, image_width])

        points_3d_2d, flag = project_point_dict[filename]
        # points_3d_2d = adjust_overlapping_points(points_3d_2d)
        flags.append(flag)

        # Define the points for perspective transformation
        h, w = mask_resized.shape[1:3]
        points_2d = [[0, 0], [w, 0], [w, h], [0, h]]

        # Perform perspective transform for the current image using the resized mask
        transformed_mask_resized = TF.perspective(mask_resized, startpoints=points_2d, endpoints=points_3d_2d)

        if is_binary:
            # Convert to grayscale and apply threshold
            transformed_mask_resized_gray = transformed_mask_resized.mean(dim=0, keepdim=True)
            transformed_mask_resized_binary = (transformed_mask_resized_gray > 0.02).float()

            # Merge channels
            transformed_mask_resized = transformed_mask_resized_binary.repeat(3, 1, 1)

        # Resize and pad the transformed mask
        transformed_mask_resized_padded = resize_and_pad(transformed_mask_resized.unsqueeze(0), img_size)

        transformed_masks.append(transformed_mask_resized_padded.squeeze(0))

    # Stack all transformed masks into a single tensor
    # transformed_masks_unpadded = torch.stack(transformed_masks_unpadded, dim=0)
    transformed_masks = torch.stack(transformed_masks, dim=0)

    return transformed_masks, flags

def project_mask_pytorch_batch_worigin(vehicle_images, filenames, mask, is_binary, project_point_dict, img_size):
    """
    :param vehicle_images: Tensor of shape [N, C, H, W]
    :param filenames: list of str
    :param mask: Tensor of shape [C, H, W]
    :param is_binary: bool
    :param project_point_dict: dict
    :param img_size: int, target size for padding
    :return Tuple(Tensor [N, C, img_size, img_size], List[flags])
    """
    transformed_masks = []
    transformed_masks_unpadded = []
    flags = []

    for i in range(len(vehicle_images)):
        vehicle_image = vehicle_images[i]
        filename = filenames[i]

        # Get the size of the current vehicle image
        image_height, image_width = vehicle_image.shape[1:3]

        # Resize mask to match the size of the current vehicle image
        mask_resized = TF.resize(mask, [image_height, image_width])

        points_3d_2d, flag = project_point_dict[filename]
        # points_3d_2d = adjust_overlapping_points(points_3d_2d)
        flags.append(flag)

        # Define the points for perspective transformation
        h, w = mask_resized.shape[1:3]
        points_2d = [[0, 0], [w, 0], [w, h], [0, h]]

        # Perform perspective transform for the current image using the resized mask
        transformed_mask_resized = TF.perspective(mask_resized, startpoints=points_2d, endpoints=points_3d_2d)

        if is_binary:
            # Convert to grayscale and apply threshold
            transformed_mask_resized_gray = transformed_mask_resized.mean(dim=0, keepdim=True)
            transformed_mask_resized_binary = (transformed_mask_resized_gray > 0.02).float()

            # Merge channels
            transformed_mask_resized = transformed_mask_resized_binary.repeat(3, 1, 1)

        transformed_masks_unpadded.append(transformed_mask_resized)

        # Resize and pad the transformed mask
        transformed_mask_resized_padded = resize_and_pad(transformed_mask_resized.unsqueeze(0), img_size)

        transformed_masks.append(transformed_mask_resized_padded.squeeze(0))

    # Stack all transformed masks into a single tensor
    # transformed_masks_unpadded = torch.stack(transformed_masks_unpadded, dim=0)
    transformed_masks = torch.stack(transformed_masks, dim=0)

    return transformed_masks, transformed_masks_unpadded, flags


def showtensor(img_tensor):
    img_tensor_display = img_tensor.detach().permute(1, 2, 0).cpu()

    image = img_tensor_display.numpy()
    plt.imshow(image)
    plt.show()

