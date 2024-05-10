import os
import numpy as np
import torch, random
from monai.data import MetaTensor
from typing import Tuple, List, Dict, Union
from monai.transforms import (
    Compose,
    CropForegroundd,
    ScaleIntensityRangePercentilesd,
    Lambdad,
    AsDiscrete,
    LoadImaged,
    MapLabelValued,
    ResizeWithPadOrCropd,
    Flipd,
    Identityd,
    SaveImaged,
    CenterSpatialCropd,
    ScaleIntensityRanged,
)
from PIL import Image
from utils.tools import check_mk_dirs, check_all_exist


def split_and_save(data: dict[MetaTensor], keys: List[str], crop_size: List[int], output_dir: str, clip_z=False):
    """
    Splits the given dictionary `data` containing multiple tensors along the y-dimension 
    into two halves ("left" and "right") and saves them separately.

    Args:
        data: A dictionary containing tensors (e.g., image, mask) as key-value pairs.
        keys: A list of keys (strings) specifying the tensors to be split and saved.
        crop_size: A list of integers representing the original image size (width, height).
        output_dir: The directory path where the split images will be saved.
        clip_z: A boolean flag (default: False). If True, it clips the saved images along 
                the z-dimension based on the non-zero entries in the "mask" tensor.

    Returns:
        The modified dictionary `data` with potentially clipped tensors.
    """
    half_width = crop_size[0] // 2
    for i, side in enumerate(["L", "R"]):
        transform = Compose(  # crop half and flip first
            [
                Lambdad(keys, lambda x: x[:, half_width * i : half_width * (i + 1)]),  # crop half on x
                Flipd(keys, [0]) if i == 1 else Identityd(keys),  # mirrow right side
            ]
        )
        _data = transform(data)
        if clip_z and _data["mask"].numel() > 0 and _data["mask"].max() > 0:
            index = torch.where(_data["mask"] > 0)[-1]
            z_min, z_max = index.min(), index.max()
            for k in keys:
                _data[k] = _data[k][..., z_min : z_max + 1]
        for key in keys:  # save by key, to add postfix
            image_saver = SaveImaged(
                keys=key,
                output_dir=output_dir,
                output_postfix=side + "_" + key,
                resample=False,
                separate_folder=False,
            )
            _data = image_saver(_data)


def get_ct_nii(input_dir: str, case_file: str, output_dir: str, crop_size: list) -> Tuple[str]:
    """Generate and return a dictionary of file paths for CT and mask images.

    This function checks if the CT and mask images for a given case have already been generated.
    If not, it processes the images using the specified transformations and saves them to the output directory.

    Args:
        input_dir (str): Path to the input directory containing the original CT and mask images.
        case_file (str): File name of the case (without the .nii.gz extension).
        output_dir (str): Path to the output directory where the processed CT and mask images will be saved.
        crop_size (list): List containing the size to which the images should be cropped.

    Returns:
        Tuple[str]: A dictionary containing the file paths for the CT and mask images for both left and right sides.
    """
    file_dicts = []
    generated = True
    for side in ["L", "R"]:
        image_dir = output_dir / f"{case_file}_{side}_image.nii.gz"
        mask_dir = output_dir / f"{case_file}_{side}_mask.nii.gz"
        file_dicts.append({"image": image_dir, "mask": mask_dir})
        if not (os.path.exists(image_dir) and os.path.exists(mask_dir)):
            generated = False

    if generated:
        return file_dicts
    
    transform = Compose(
        [
            LoadImaged(["image", "mask"], image_only=True, ensure_channel_first=True),
            ScaleIntensityRanged("image", -300, 500, 0, 1, clip=True),  # adjust ct window
            MapLabelValued("mask", [2, 4, 3], [0, 0, 1]),  # filter left and right internal carotid artery
            CropForegroundd(["image", "mask"], source_key="image"),
            ResizeWithPadOrCropd(["image", "mask"], crop_size),
        ]
    )
    data = transform(
        {
            "image": input_dir / "image" / (case_file + '.nii.gz'),
            "mask": input_dir / "mask" / (case_file + '.nii.gz'),
        }
    )
    split_and_save(data, ["image", "mask"], crop_size, output_dir, clip_z=True)
    return file_dicts


def get_mri_nii(input_dir: str, case_file: str, output_dir: str, crop_size: list) -> Tuple[str]:
    """Generate processed MRI NII files and return list of directories for Monai dataset.

    This function checks if the MRI NII files for a given case have already been generated.
    If not, it processes the images using the specified transformations and saves them to the output directory.

    Args:
        input_dir (str): Path to the input directory containing the original MRI NII files.
        case_file (str): File name of the case (without the .nii.gz extension).
        output_dir (str): Path to the output directory where the processed MRI NII files will be saved.
        crop_size (list): List containing the size to which the images should be cropped.

    Returns:
        tuple[str]: A list of directories containing the file paths for the processed MRI NII files for both left and right sides.
    """
    file_dicts = []
    for side in ["L", "R"]:
        file_dicts.append(
            {
                "image": output_dir / f"{case_file}_{side}_image.nii.gz",
                "mask": output_dir / f"{case_file}_{side}_mask.nii.gz",
                "sam": output_dir / f"{case_file}_{side}_sam.nii.gz",
                "interp": output_dir / f"{case_file}_{side}_interp.nii.gz",
            }
        )
    if check_all_exist(file_dicts[0].values()) and check_all_exist(file_dicts[1].values()):  # check all files processed?
        return file_dicts
    
    transform = Compose(
        [
            LoadImaged(["image", "mask", "sam", "interp"], image_only=True, ensure_channel_first=True),
            ScaleIntensityRangePercentilesd(["image"], 0, 98, 0, 1, clip=True),
            CropForegroundd(["image", "mask", "sam", "interp"], source_key="image"),
            CenterSpatialCropd(["image", "mask", "sam", "interp"], crop_size),
        ]
    )
    data = transform(
        {
            "image": input_dir / f"{case_file}_image.nii.gz",
            "mask": input_dir / f"{case_file}_mask.nii.gz",
            "sam": input_dir / f"{case_file}_sam.nii.gz",
            "interp": input_dir / f"{case_file}_interp.nii.gz",
        }
    )
    for key in ["image", "mask", "sam", "interp"]:
        data[key].meta["filename_or_obj"] = case_file  # format file names
    split_and_save(data, ["image", "mask", "sam", 'interp'], crop_size, output_dir, clip_z=True)
    return file_dicts


def get_mri_pngs(input_dir: str, case_file: str, output_dir: str, crop_size: list) -> Tuple[str]:
    """Generate processed MRI PNG files and return list of directories for Monai dataset.

    This function checks if the MRI PNG files for a given case have already been generated.
    If not, it processes the images using the specified transformations and saves them to the output directory.

    Args:
        input_dir (str): Path to the input directory containing the original MRI NII files.
        case_file (str): File name of the case (without the .nii.gz extension).
        output_dir (str): Path to the output directory where the processed MRI PNG files will be saved.
        crop_size (list): List containing the size to which the images should be cropped.

    Returns:
        Tuple[str]: A list of directories containing the file paths for the processed MRI PNG files for both left and right sides.
    """
    case_dir = check_mk_dirs(output_dir / case_file)
    if os.listdir(case_dir):  # any file existed
        image_list = list(case_dir.glob("*_image.png"))
        file_dicts = [{"image": i, "mask": i.parent / str(i.name).replace('_image', '_mask')} for i in image_list]
        return file_dicts
    
    transform = Compose(
        [
            LoadImaged(["image", "mask"], image_only=True, ensure_channel_first=True),
            ScaleIntensityRangePercentilesd(["image"], 0, 98, 0, 1, clip=True),
            CropForegroundd(["image", "mask"], source_key="image"),
            ResizeWithPadOrCropd(["image", "mask"], crop_size),
        ]
    )
    data = transform(
        {
            "image": input_dir / f"{case_file}_image.nii.gz",
            "mask": input_dir / f"{case_file}_mask.nii.gz",
        }
    )
    step = crop_size[0] // 2
    file_dicts = []
    for idx, side in enumerate(["L", "R"]):
        image = data["image"][:, step * idx : step * (idx + 1)]
        mask = data["mask"][:, step * idx : step * (idx + 1)]
        mask[mask == 3] = 2 # COSMOS
        annotated_slices = torch.unique(torch.where(mask > 0)[-1])
        for anno_id in annotated_slices:
            mask_slice = AsDiscrete(to_onehot=3)(mask[..., anno_id].detach().numpy())
            image_slice = image[0, ..., anno_id].numpy()
            if idx == 1:
                image_slice, mask_slice = np.flip(image_slice, 0), np.flip(mask_slice, 1)
            image_dir = case_dir / f"{case_file}_{anno_id}{side}_image.png"
            mask_dir = case_dir / f"{case_file}_{anno_id}{side}_mask.png"
            image_slice = image_slice.T * 255
            mask_slice = mask_slice.T * 255
            Image.fromarray(image_slice.astype(np.uint8)).save(image_dir)
            Image.fromarray(mask_slice.astype(np.uint8)).save(mask_dir)
            file_dicts.append({"image": image_dir, "mask": mask_dir})
    return file_dicts


def get_ct_pngs(input_dir: str, case_file: str, output_dir: str, crop_size: Tuple[int]) -> List[Dict[str, Union[str, int]]]:
    """
    Extracts CT and mask images from a case and saves them as PNG files.

    Args:
        input_dir (str): Path to the directory containing the input data.
        case_file (str): Name of the case file.
        output_dir (str): Path to the directory where the output PNG files will be saved.
        crop_size (tuple): Size of the cropped foreground region.

    Returns:
        A list of dictionaries containing file paths and labels for each image.
    """
    case_dir = check_mk_dirs(output_dir / case_file)
    if os.listdir(case_dir):  # any existed
        image_list = list(case_dir.glob("*_image.png"))
        file_dicts = [{"image": i, "mask": i.parent / str(i.name).replace('_image', '_mask')} for i in image_list]
        return file_dicts
    
    # Load and transform CT and mask images
    transform = Compose(
        [
            LoadImaged(["image", "mask"], image_only=True, ensure_channel_first=True),
            ScaleIntensityRanged("image", -300, 500, 0, 1, clip=True),
            MapLabelValued("mask", [2, 4, 3], [0, 0, 1]),
            CropForegroundd(
                ["image", "mask"], source_key="mask", margin=[crop_size[0] // 2, 8, crop_size[2] // 2]
            ),  # mainly crop on y axis
            ResizeWithPadOrCropd(["image", "mask"], crop_size),
        ]
    )
    data = transform(
        {
            "image": input_dir / "image" / (case_file + '.nii.gz'),
            "mask": input_dir / "mask" / (case_file + '.nii.gz'),
        }
    )
    step = crop_size[0] // 2
    file_dicts = []
    for idx, side in enumerate(["L", "R"]):
        image, mask = (
            data["image"][:, step * idx : step * (idx + 1)],
            data["mask"][:, step * idx : step * (idx + 1)],
        )
        annotated_slices = torch.unique(torch.where(mask > 0)[-1])
        if len(annotated_slices) > 50:  # TODO: remove
            annotated_slices = random.sample(list(annotated_slices), 50)
        for anno_id in annotated_slices:
            mask_slice = AsDiscrete(to_onehot=3)(mask[..., anno_id].numpy())
            image_slice = image[0, ..., anno_id].numpy()
            if idx == 1:
                image_slice, mask_slice = np.flip(image_slice, 0), np.flip(mask_slice, 1)

            image_dir = case_dir / f"{case_file}_{anno_id}{side}_image.png"
            mask_dir = case_dir / f"{case_file}_{anno_id}{side}_mask.png"

            image_slice = image_slice.T * 255
            mask_slice = mask_slice.T * 255
            Image.fromarray(image_slice.astype(np.uint8)).save(image_dir)
            Image.fromarray(mask_slice.astype(np.uint8)).save(mask_dir)
            file_dicts.append({"image": image_dir, "mask": mask_dir})
    return file_dicts
