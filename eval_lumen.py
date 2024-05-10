import argparse
import os
from pathlib import Path

import numpy as np
import torch
from monai.data import MetaTensor
from monai.metrics import DiceMetric
from monai.metrics.hausdorff_distance import compute_hausdorff_distance
from monai.metrics.surface_distance import compute_average_surface_distance
from monai.transforms import Compose, FillHoles, LoadImage, RemoveSmallObjects, ResizeWithPadOrCrop, SaveImage
from monai.utils.misc import set_determinism
from torchmetrics import Precision, Recall

from models import define_model
from utils.predict import avoid_nan, clDice, merge_split, setup_case, shortest_path
from utils.tools import filtering_mask, load_configs, load_split


def main():
    set_determinism(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--ckpt", default="dou_cosmos_f0", type=str)
    parser.add_argument("-n", "--name", default="last.ckpt")
    parser.add_argument("-p", "--predict", default=True, help='turn off this to simply compute existed predictions')
    parser.add_argument("-d", "--device", default="cuda:0", type=str)
    args = parser.parse_args()
    print(args.ckpt)
    root_dir = Path(os.getenv("DATA_ROOT")) 
    lumen_save = root_dir / "VWS_exp" / "pred_lumen" / args.ckpt
    cfg = load_configs(root_dir / "VWS_exp" / args.ckpt, "cfg")

    input_dir = root_dir / cfg.dataset / "preprocessed" / "mri_nii_raw"
    _, _, test_cases = load_split(root_dir / cfg.dataset, cfg.fold)

    if args.predict:
        model = define_model(cfg, init=False)
        model = model.load_from_checkpoint(root_dir / "VWS_exp" / args.ckpt / args.name, strict=False, cfg=cfg)
        model.to(args.device)
        model.eval()

    eval_list = []
    sorted(test_cases)
    image_saver = SaveImage(lumen_save, output_postfix="", resample=True, separate_folder=False, print_log=False)
    for case_file in test_cases:
        file_dicts = setup_case(case_file, root_dir, cfg)
        if args.predict:
            lumen_pred = predict_case(case_file, list(file_dicts), cfg, model, input_dir, args.device)
            image_saver(lumen_pred)
        spacing = 0.35 if cfg.dataset == "careII" else 0.6
        result = evaluate_case(case_file, input_dir, lumen_save, spacing)
        if result != None:
            eval_list.append(result)

    print("Average Evaluation:")
    for key in eval_list[0].keys():
        print(f"{key}: {np.mean([d[key] for d in eval_list]): .3f}", end="  ")
    print("\n")


def predict_case(case_file, file_dicts, cfg, model, input_dir, device):
    """
    Predict the case using the provided model and configuration.

    Args:
        case_file (str): The name of the case file.
        file_dicts (list): A list of file dictionaries containing image data.
        cfg (object): The configuration object containing necessary parameters.
        model (object): The trained model to be used for prediction.
        input_dir (str): The directory where the input case file is located.
        device (torch.device): The device to run the prediction on.

    Returns:
        object: The predicted result for the case.
    """
    pred_list = []
    for i in range(2):
        with torch.no_grad():
            image = file_dicts[i]["image"].to(device)
            lumen_pred = model(image.unsqueeze(0))[0].cpu()

        post_trans = Compose([FillHoles(1), RemoveSmallObjects(min_size=5)])  # slice-wise removal
        for d in range(lumen_pred.shape[-1]):  # slice-wise
            lumen_pred[..., d] = post_trans(lumen_pred[..., d])
        lumen_pred = lumen_pred.to(torch.float32)
        center_list, lumen_pred = shortest_path(lumen_pred)
        pred_list.append(lumen_pred)
    lumen_pred = merge_split(pred_list)
    lumen_pred[lumen_pred > 1] = 1
    cropped_image = Compose(
        [
            LoadImage(image_only=True, ensure_channel_first=True),
            ResizeWithPadOrCrop(cfg.mri_crop_size),
        ]
    )(os.path.join(input_dir, case_file + "_image.nii.gz"))
    lumen_pred = MetaTensor(lumen_pred, meta=cropped_image.meta)
    return lumen_pred


def evaluate_case(case_file, input_dir, pred_dir, spacing):
    """
    Evaluate the performance of the model's prediction for a specific case.

    Args:
        case_file (str): The name of the case file.
        input_dir (str): The directory where the input files are located.
        pred_dir (str): The directory where the predicted output files are located.

    Returns:
        dict: A dictionary containing the evaluation results, including dice, cldice, hd, and f1.
    """
    image_reader = LoadImage(image_only=True, ensure_channel_first=True)

    mask = image_reader(os.path.join(input_dir, case_file + "_mask.nii.gz"))
    sam_mask = image_reader(os.path.join(input_dir, case_file + "_sam.nii.gz"))
    lumen_pred = image_reader(os.path.join(pred_dir, case_file + "_image.nii.gz"))
    mask = (mask == 1).to(torch.float32)

    dice_metric = DiceMetric(include_background=False, reduction="mean", ignore_empty=False)
    hd_metric = []
    sd_metric = []
    cldice_metric = []
    recall_metric = Recall(task="binary", average="macro")
    precision_metric = Precision(task="binary", average="macro")

    for i in range(2):
        step = mask.shape[1] // 2
        oneside_mask = mask[:, step * i : step * (i + 1)].int()
        oneside_sam = sam_mask[:, step * i : step * (i + 1)].int()
        oneside_pred = lumen_pred[:, step * i : step * (i + 1)].int()
        non_zeros, filter_mask = filtering_mask(oneside_mask)
        if non_zeros.any():
            for idx in non_zeros:
                mask_slice = oneside_mask[None, ..., idx]  # add batch dim
                pred_slice = oneside_pred[None, ..., idx]  # add batch dim
                dice_metric(y_pred=pred_slice, y=mask_slice)
                recall_metric(pred_slice, mask_slice)
                precision_metric(pred_slice, mask_slice)
                hd = compute_hausdorff_distance(pred_slice, mask_slice, spacing=spacing)
                hd_metric.append(hd.mean())
                sd = compute_average_surface_distance(pred_slice, mask_slice, spacing=spacing)
                sd_metric.append(sd.mean())
            # compute clDice on the continous but labeled area (interp label could work as a filter)
            zmin, zmax = non_zeros.min(), non_zeros.max()
            cldice = clDice(
                oneside_pred[0, ..., zmin : zmax + 1], oneside_sam[0, ..., zmin : zmax + 1]
            )  # cldice without batch dim
            cldice_metric.append(cldice)
    if len(sd_metric) > 0:
        dice = dice_metric.aggregate().item()
        hd_metric = avoid_nan(hd_metric)
        sd_metric = avoid_nan(sd_metric)
        cldice_metric = avoid_nan(cldice_metric)
        hd = sum(hd_metric) / len(hd_metric)
        sd = sum(sd_metric) / len(sd_metric)
        cldice = sum(cldice_metric) / len(cldice_metric)
        recall = recall_metric.compute().item()
        precision = precision_metric.compute().item()
        print(
            f"({case_file}) Dice: %.3f, ASD: %.3f, HD: %.3f, Recall: %.3f, Precision: %.3f, clDice: %.3f"
            % (dice, sd, hd, recall, precision, cldice)
        )
        return {
            "dice": dice,
            "asd": sd,
            "hd95": hd,
            "recall": recall,
            "precision": precision,
            "cldice": cldice,
        }
    else:
        return None


if __name__ == "__main__":
    main()
