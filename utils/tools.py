import json
import os
import platform
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import torch
import yaml
from monai.data import CacheDataset, ThreadDataLoader
from monai.transforms import ScaleIntensity
from PIL import Image
from torchvision.utils import draw_segmentation_masks


def filter_invalid(dataset_dir: Path, file_dicts: List[Dict[str, Path]]) -> List[Dict[str, Path]]:
    """
    Filters out files from `file_dicts` that are listed in the `invalid.txt` file in the `dataset_dir`.

    Args:
        dataset_dir: A Path object pointing to the directory containing the dataset.
        file_dicts: A list of dictionaries containing information about each file in the dataset.

    Returns:
        A list of dictionaries containing information about each valid file in the dataset.
    """
    try:
        with open(dataset_dir / "invalid.txt", "r", encoding="utf-8") as file:
            lines = file.readlines()
        invalid = [line.strip() for line in lines]
        file_dicts = [file for file in file_dicts if file["image"].name not in invalid]
        return file_dicts
    except FileNotFoundError:
        print(f"File invalid.txt not found.")
        return []


def init_configs(path: str, args: SimpleNamespace) -> SimpleNamespace:
    """
    Initialize configuration options for the program by loading values from a YAML file
    and combining them with command-line arguments.

    Args:
        path: A string specifying the directory path where the YAML file is located.
        args: A SimpleNamespace object containing command-line arguments.

    Returns:
        A SimpleNamespace object containing the final configuration options,
        including those from the YAML file and any overrides from command-line arguments.
    """
    with open(os.path.join(path, f"{args.config}.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = SimpleNamespace(**cfg)
    if platform.system().lower() == "windows":
        cfg.root_dir = "d:/data/"  # TODO: change to your own path if using windows
    else:
        cfg.root_dir = os.getenv("DATA_ROOT")
    cfg.resume = args.resume
    cfg.model_name = args.model_name
    cfg.fold = args.fold
    cfg.log_dir = check_mk_dirs(os.path.join(cfg.root_dir, f"VWS_exp", args.model_name))
    return cfg


def load_configs(path: str = ".", name: str = "cfg", resume=None) -> SimpleNamespace:
    """
    Load configuration from a YAML file into a SimpleNamespace object.

    Args:
        path (str): Path to the directory where the configuration file is located.
            Defaults to the current directory.
        name (str): Name of the configuration file without the '.yaml' extension.
            Defaults to 'cfg'.
        resume (str, optional): Path to a different configuration file to load in
            case of resuming a previous run.

    Returns:
        SimpleNamespace: A SimpleNamespace object containing the loaded
            configuration.
    """
    path = resume if resume else path
    with open(os.path.join(path, f"{name}.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = SimpleNamespace(**cfg)
    return cfg


def load_split(path: str, fold: int):
    """Loads the train, val, and test splits from a JSON file for a given fold.

    Args:
        path: The path to the directory where the JSON file is located.
        fold: The fold number for which to load the splits.

    Returns:
        A tuple containing three lists: train, val, and test. Each list contains the
        file paths of the images in the corresponding split.

    Raises:
        FileNotFoundError: If the JSON file does not exist in the given path.
    """
    with open(os.path.join(path, f"fold_{fold}.json"), encoding="utf-8", mode="r") as f:
        split = json.load(f)
    return split["train"], split["val"], split["test"]


def save_current_configs(cfg: SimpleNamespace) -> SimpleNamespace:
    """Save the current configuration to a YAML file.

    Args:
        cfg (SimpleNamespace): The configuration object to save.

    Returns:
        SimpleNamespace: The same configuration object that was passed in.

    Raises:
        PermissionError: If the user does not have permission to write to the log directory.

    Notes:
        The configuration is saved to a file named "cfg.yaml" in the log directory.
    """
    file_name = os.path.join(cfg.log_dir, "cfg.yaml")
    try:
        with open(file_name, "wt") as cfg_file:
            yaml.dump(cfg.__dict__, cfg_file)
    except PermissionError as error:
        print("permission error {}".format(error))
        pass
    return cfg


def check_mk_dirs(paths):
    """Create new directories if they do not exist.

    Args:
        paths (Union[str, List[str]]): The path(s) of the directory(ies) to create.
            Can be a single string or a list of strings.

    Returns:
        Union[str, List[str]]: The input path(s), unchanged.

    Notes:
        This function checks if the given directory path(s) exist. If they do not exist,
        it creates them. If the directories already exist, it does nothing.

        The function can handle input paths as both strings and lists of strings.

        Example usage:

        >>> check_mk_dirs("data")
        "data"

        >>> check_mk_dirs(["data", "images"])
        ["data", "images"]
    """
    if isinstance(paths, list):
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
    else:
        if not os.path.exists(paths):
            os.makedirs(paths, exist_ok=True)
    return paths


def check_all_exist(file_list: list[str]):
    """Check whether all paths in `file_list` exist already.

    Args:
        file_list (list[str]): A list of file paths.

    Returns:
        bool: True if all file paths exist, False otherwise.

    Notes:
        This function checks if all file paths in the provided list exist in the
        file system. If any of the paths does not exist, the function returns False.

        Example usage:

        >>> check_all_exist(["data/file1.txt", "data/file2.txt"])
        True

        >>> check_all_exist(["data/file1.txt", "data/nonexistent.txt"])
        False
    """
    for file in file_list:
        if not os.path.exists(file):
            return False
    return True


def get_dataloader(files, trans, cfg, shuffle: bool):
    """
    Generate a PyTorch DataLoader for the given file directories, Monai transforms,
    and configuration options.

    Args:
        files (list): A list of file paths or directories to be included in the dataset.
        trans (nn.Module): A Monai transform object to apply to the images.
        cfg (SimpleNamespace): A SimpleNamespace object containing configuration options.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.

    Returns:
        DataLoader: A PyTorch DataLoader object.

    Notes:
        This function creates a PyTorch DataLoader object for the given file directories,
        Monai transforms, and configuration options. The DataLoader is created using the
        ThreadDataLoader class, which is a subclass of the PyTorch DataLoader that
        provides support for multi-threading.

        The CacheDataset class is used to wrap the dataset, which provides support for
        caching of images to improve performance. The cache_rate option is set based on
        the value of the cache\_rate attribute in the cfg object.
    """
    dataset = CacheDataset(data=files, transform=trans, cache_rate=cfg.cache_rate, num_workers=cfg.num_workers)
    dataloader = ThreadDataLoader(dataset, batch_size=cfg.batch_size, shuffle=shuffle, num_workers=0)
    return dataloader


def filtering_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    Filter a 3D mask by removing background and getting annotated slice indices.

    Args:
        mask (torch.Tensor): A 3D tensor of shape (C, H, W, D) representing a mask.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:

            - non_zero (torch.Tensor): A 1D tensor of unique indices corresponding to annotated slices.
            - filter_mask (torch.Tensor): A 3D tensor of shape (1, H, W, D) with ones at annotated slice locations.
    """
    C, H, W, D = mask.shape
    if C > 1:
        mask = mask[1:]  # remove background
    filter_mask = torch.zeros((1, H, W, D)).type_as(mask)  # 1, ...
    if mask.max() > 0.0:
        non_zero = torch.where(mask > 0)[-1].unique()
        filter_mask[..., non_zero] = 1
    else:
        non_zero = torch.tensor([], dtype=torch.long)
    return non_zero.to(mask.device), filter_mask


def save_segmentation_result(dim: int, image: torch.Tensor, masks: list[torch.Tensor], save_dir: str, alpha=0.5):
    """Save segmentation results with comparison to the ground truth.

    Args:
        dim (int): The number of dimensions of the input image. Should be either 2 or 3.
        image (torch.Tensor): The input image tensor.
        masks (list[torch.Tensor]): A list of mask tensors.
        save_dir (str): The directory to save the result image.
        alpha (float, optional): The transparency of the mask. Defaults to 0.5.

    Raises:
        AssertionError: If `dim` is not 2 or 3.

    Notes:
        This function saves the segmentation results as an image with the given directory.
        The result image will have the same dimensions as the input image.

        If `dim` is 2, the function maps each mask in `masks` to a color and overlays
        the masks onto the input image with the given `alpha` transparency. The result
        image will have 3 color channels.

        If `dim` is 3, the function maps each mask in `masks` to a color and overlays
        the masks onto the input image along the last dimension with the given `alpha`
        transparency. The result image will have the same shape as the input image.

        The function assumes that the input image and the masks have the same shape.
    """
    assert dim in [2, 3], "Unsupported dim option!"
    image = image.repeat([3] + [1] * dim).cpu()
    if masks[0].shape[0] > 1:
        colors = None
        masks = [(mask > 0.5).cpu() for mask in masks]
    else:
        colors = ["#D2691E"]
        masks = [(mask > 0.5).cpu() for mask in masks]
    image = ScaleIntensity(0, 255)(image).to(torch.uint8)

    if dim == 2:
        blended_masks = [draw_segmentation_masks(image, mask, alpha=alpha, colors=colors) for mask in masks]
        visual_image = torch.cat(blended_masks, 2).numpy().astype(np.uint8)  # W, H, C
        Image.fromarray(visual_image.T).save(save_dir)
    else:
        blended_masks = [
            torch.stack(
                [
                    draw_segmentation_masks(image[..., d], mask[..., d], alpha=alpha, colors=colors)
                    for d in range(image.shape[-1])
                ]
            )
            for mask in masks
        ]
        visual_image = torch.cat(blended_masks, 2).numpy().astype(np.uint8)  # D, C, H, W
        img_list = [Image.fromarray(v.T) for v in visual_image]
        img_list[0].save(save_dir, save_all=True, append_images=img_list[1:])
