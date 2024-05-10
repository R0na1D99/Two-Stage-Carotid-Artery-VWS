import os

import numpy as np
import torch

from monai.transforms import (
    Compose,
    EnsureType,
    LoadImaged,
    EnsureTyped,
    Lambdad,
    ScaleIntensityRangePercentilesd,
    Flipd,
    Identityd,
    ResizeWithPadOrCropd,
)
from typing import List, Union
from skimage.morphology import skeletonize_3d
from scipy.ndimage import center_of_mass, label as scipy_label
from pathlib import Path
from types import SimpleNamespace

def setup_case(case_file: str, root_dir: Path, cfg: SimpleNamespace, lumen_dir=None):
    """Set up a single case with transforms.

    Args:
        case_file (str): The name of the case.
        root_dir (Path): The root directory of the data.
        cfg (Namespace): The configuration object.
        lumen_dir (Optional[Path], optional): The directory containing the lumen files. Defaults to None.

    Returns:
        List[Dict[str, torch.Tensor]]: A list of file dictionaries with the applied transforms.
    """
    input_dir = root_dir / cfg.dataset / "preprocessed" / "mri_nii_raw"
    half_width = cfg.mri_crop_size[0] // 2
    file_dicts = []
    for i in range(2):
        transform = Compose(
            [
                LoadImaged(["image", "mask", 'lumen'], image_only=True, ensure_channel_first=True),
                ScaleIntensityRangePercentilesd(["image"], 0, 98, 0, 1, clip=True),
                ResizeWithPadOrCropd(["image", "mask", 'lumen'], cfg.mri_crop_size),
                Lambdad(["image", "mask", 'lumen'], lambda x: x[:, half_width * i : half_width * (i + 1)]),  # crop half on x
                Flipd(["image", "mask", 'lumen'], [0]) if i == 1 else Identityd(["image", "mask"]),  # mirrow right side
                EnsureTyped(["image", "mask", 'lumen'], dtype=torch.float32, track_meta=False)
            ]
        )
        lumen = lumen_dir / f'{case_file}_image.nii.gz' if lumen_dir else input_dir / f'{case_file}_image.nii.gz'

        file_dict = transform(
            {
                'image': input_dir / f'{case_file}_image.nii.gz',
                'mask': input_dir / f'{case_file}_mask.nii.gz',
                'lumen': lumen
            }
        )
        file_dicts.append(file_dict) 
    return file_dicts

def avoid_nan(result: List) -> List:
    """Remove NaN or Inf values from a list of tensors.

    Args:
        result (list): A list of tensors.

    Returns:
        list: A new list of tensors with NaN or Inf values removed.

    Notes:
        This function iterates over the input list and checks if each element is a tensor.
        If it is not, it converts the element to a tensor. Then, it checks if the tensor
        contains NaN or Inf values. If it does, the tensor is removed from the list.
        Otherwise, the tensor is added to a new list. Finally, the new list is returned.
    """
    new_list = []
    for i in result:
        if not isinstance(i, torch.Tensor):
            i = torch.as_tensor(i)
        if torch.isnan(i) or torch.isinf(i):
            continue
        else:
            new_list.append(i)
    return new_list

def betti_number(mask: np.ndarray) -> float:
    """Calculate the Betti number of a binary image.

    The Betti number is a topological invariant that counts the number of holes
    in a binary image. This function calculates the Betti number of a binary image
    represented as a NumPy array.

    Args:
        mask (np.ndarray): A binary image represented as a NumPy array.

    Returns:
        float: The Betti number of the input image.

    Notes:
        This function uses the SciPy library to label the connected components
        of the input image and calculates the Betti number as the number of
        holes in the image.
    """
    mask = np.asarray(mask).astype(np.int8)
    _, num_components = scipy_label(1 - mask) # num holes
    betti_1 = num_components - 1  
    return float(betti_1)


def shortest_path(mask: np.ndarray) -> List[np.ndarray]:
    """Find a shortest path by connecting closest connected components from top to bottom.

    This function takes a 4D numpy array of binary values and returns a list
    containing the centers of the connected components along the shortest path
    and a 4D numpy array of the same shape as the input `mask` with the shortest
    path connected components.

    Args:
        mask (np.ndarray): A 4D numpy array of binary values.

    Returns:
        List[np.ndarray]: A list containing the centers of the connected components
        along the shortest path and a 4D numpy array of the same shape as the input
        `mask` with the shortest path connected components.

    Notes:
        This function uses the `label` function from the scipy library to
        label the connected components in the input `mask`. It then calculates
        the centers of mass of each connected component and finds the shortest
        path by connecting the closest components from top to bottom.
    """
    mask = np.asarray(mask)
    C, H, W, D = mask.shape
    indice = np.unique(np.where(mask > 0)[-1])  # Get labeled slices idx from 2 channels
    center_matrix = np.zeros((len(indice), 3, 3))
    component_matrix = np.zeros((len(indice), 3, H, W))
    for d, slice_idx in enumerate(indice):
        lumen_mask_slice = mask[0, ..., slice_idx]
        lumen_mask_slice, num_components = scipy_label(lumen_mask_slice)
        for i in range(3):  # top 3 masks
            component = lumen_mask_slice == i + 1
            if np.any(component):
                center = center_of_mass(component)
                center_matrix[d, i] = np.array([center[0], center[1], slice_idx])
            else:
                center_matrix[d, i] = np.array([0, 0, 0])
            component_matrix[d, i] = np.float32(component)

    center_matrix = np.flip(center_matrix, 0)  # from top of brain to neck
    component_matrix = np.flip(component_matrix, 0)
    voxel_center = np.mean(center_matrix[:, 0], axis=0)  # 3,
    shortest_centers = [np.array([voxel_center[0], voxel_center[1], center_matrix[0][0][-1]])]
    shortest_components = [np.zeros_like(component_matrix[0][0])]
    for i in range(len(center_matrix)):
        current_center = shortest_centers[-1]
        start = i
        end = min(i + 50, len(center_matrix))
        centers_next_50 = center_matrix[start:end]
        compos_next_50 = component_matrix[start:end]
        distances = np.sum((centers_next_50 - current_center[None, None, ...]) ** 2, axis=-1)  # 50, 3
        closest_idx = np.array(np.where(distances == distances.min())).T[0]
        shortest_centers.append(centers_next_50[closest_idx[0], closest_idx[1]])
        shortest_components.append(compos_next_50[closest_idx[0], closest_idx[1]])

    shortest_mask = np.zeros_like(mask)
    for center, comp in zip(shortest_centers, shortest_components):
        shortest_mask[0, ..., int(center[-1])] += comp
    shortest_centers = np.flip(np.array(shortest_centers), 0)

    return shortest_centers, shortest_mask


def fix_missing_centers(centers: np.ndarray) -> List[np.ndarray]:
    """Use interpolation to predict missing slice centers.

    Args:
        centers (np.ndarray): A 2D numpy array of slice centers.

    Returns:
        List[np.ndarray]: A list of 2D numpy arrays containing the updated
        slice centers, with any missing slices interpolated.

    Notes:
        This function takes a 2D numpy array of slice centers as input and
        interpolates any missing slices to produce a list of updated slice
        centers. It is assumed that the input `centers` array is sorted by
        the slice index.
    """
    idx = 0
    new_centers = []
    for idx in range(len(centers) - 1):
        new_centers.append(centers[idx].astype(np.int32))
        ideal_next = int(centers[idx][-1] + 1)
        real_next = int(centers[idx + 1][-1])
        if real_next != ideal_next:
            step = (centers[idx + 1] - centers[idx]) / (real_next - ideal_next)
            for i in range(ideal_next, real_next):
                new_center = centers[idx] + (i - ideal_next + 1) * step
                new_center[-1] = i
                new_centers.append(np.int32(new_center))
    return new_centers

def merge_split(mask_list: List[torch.Tensor]):
    """
    Merge two splitted masks into one mask by flipping one of them.

    Args:
        mask_list (List[torch.Tensor]): A list containing two mask tensors.

    Returns:
        torch.Tensor: The merged mask tensor.
        
    Raises:
        ValueError: If mask_list does not contain exactly two tensors.
    """
    if len(mask_list) != 2:
        raise ValueError("mask_list must contain exactly two tensors.")
    
    oneside, theother = mask_list
    to_tensor = EnsureType("tensor", track_meta=False)
    oneside, theother = to_tensor(oneside), to_tensor(theother)
    mask = torch.cat([oneside, theother.flip(1)], 1)
    return mask



def clDice(v_p, v_l):
    """cldice metric"""
    temp = 1e-5
    v_p, v_l = np.asarray(v_p), np.asarray(v_l)
    cl_score = lambda v, s: np.sum(v * s) / (np.sum(s) + temp)
    tprec = cl_score(v_p, skeletonize_3d(v_l))
    tsens = cl_score(v_l, skeletonize_3d(v_p))
    return 2 * tprec * tsens / (tprec + tsens + temp)
