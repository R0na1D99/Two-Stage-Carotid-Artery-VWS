import argparse
import os
from pathlib import Path

import numpy as np
import torch
from monai.data import MetaTensor
from monai.metrics.surface_distance import compute_average_surface_distance
from monai.metrics.hausdorff_distance import compute_hausdorff_distance
from monai.metrics import DiceMetric 
from torchmetrics import Recall, Precision
from monai.transforms import (
    Compose,
    GaussianSmooth,
    LabelFilter,
    LoadImage,
    ResizeWithPadOrCrop,
    SaveImage,
    ScaleIntensity,
)
from monai.utils.misc import set_determinism
from models import define_model
from utils.predict import merge_split, setup_case, avoid_nan, betti_number
from utils.tools import filtering_mask, load_configs, load_split


def main():
    set_determinism(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--ckpt", default="wall_2c_cosmos_f1", type=str)
    parser.add_argument("-l", "--lumen", default="dou_cosmos_f1", type=str)
    parser.add_argument("-d", "--device", default="cuda:0", type=str)
    parser.add_argument("-p", "--predict", default=1, type=int)
    args = parser.parse_args()

    root_dir = Path(os.getenv("DATA_ROOT"))
    if args.lumen:
        lumen_dir = root_dir / "VWS_exp" / "pred_lumen" / args.lumen
    else:
        lumen_dir = None
    wall_save = root_dir / "VWS_exp" / "pred_wall" / args.ckpt
    cfg = load_configs(root_dir / "VWS_exp" / args.ckpt, "cfg")
    input_dir = root_dir / cfg.dataset / "preprocessed" / "mri_nii_raw"

    _, _, test_cases = load_split(root_dir / cfg.dataset, cfg.fold)

    if args.predict:
        model = define_model(cfg, init=False)
        model = model.load_from_checkpoint(root_dir / "VWS_exp" / args.ckpt / "last.ckpt", strict=False, cfg=cfg)
        model.to(args.device)
        model.eval()
    print(args.ckpt)

    eval_list = []
    sorted(test_cases)
    for case_file in list(test_cases):
        file_dicts = setup_case(case_file, root_dir, cfg, lumen_dir)
        if args.predict:
            pred = predict_case(case_file, list(file_dicts), cfg, model, input_dir, args.device)
            SaveImage(wall_save, output_postfix="", resample=True, separate_folder=False, print_log=False)(pred)
        spacing = 0.35 if cfg.dataset == 'careII' else 0.6
        result = evaluate_case(case_file, input_dir, wall_save, spacing)
        if result:
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
        image = file_dicts[i]["image"]
        C, H, W, D = image.shape
        for d in range(D):
            image[..., d] = ScaleIntensity()(image[..., d])
        if cfg.in_channels == 2:
            lumen_pred = file_dicts[i]["lumen"]
            lumen_pred = GaussianSmooth(sigma=cfg.sigma)(lumen_pred)  # TODO: To validate gaussian, comment this line
            inputs = torch.cat([image, lumen_pred], 0).to(device)
        elif cfg.stage == 'polar':
            lumen_pred = file_dicts[i]["lumen"]
            inputs = torch.cat([image, lumen_pred], 0).to(device)
        else:
            inputs = image.to(device)
        with torch.no_grad():
            wall_pred = model(inputs.unsqueeze(0))[0].cpu()

        pred_list.append(wall_pred)

    wall_pred = merge_split(pred_list)
    wall_pred = LabelFilter([1, 2])(wall_pred)
    cropped_image = Compose(
        [
            LoadImage(image_only=True, ensure_channel_first=True),
            ResizeWithPadOrCrop(cfg.mri_crop_size),
        ]
    )(os.path.join(input_dir, case_file + "_image.nii.gz"))
    wall_pred = MetaTensor(wall_pred, meta=cropped_image.meta) # simulate preprocess, to restore predictions to original affine
    return wall_pred


def evaluate_case(case_file, input_dir, pred_dir, spacing):
    image_reader = LoadImage(image_only=True, ensure_channel_first=True, reader='ITKReader')

    mask = image_reader(os.path.join(input_dir, case_file + "_mask.nii.gz"))
    pred = image_reader(os.path.join(pred_dir, case_file + "_image.nii.gz"))

    recall_metric = Recall(task="binary", average="macro")
    precision_metric = Precision(task="binary", average="macro")
    dice_metric = DiceMetric(include_background=True, reduction="mean", ignore_empty=False)
    hd_metric = []
    nwi_metric = []
    sd_metric = []
    bnum_metric = []

    for i in range(2):
        step = mask.shape[1] // 2
        oneside_lumen_mask = (mask == 1).float()[:, step * i : step * (i + 1)]
        oneside_lumen_pred = (pred == 1).float()[:, step * i : step * (i + 1)]
        # oneside_wall_mask = (mask >= 2).float()[:, step * i : step * (i + 1)]  
        oneside_wall_mask = (mask == 2).float()[:, step * i : step * (i + 1)] # TODO: change here to verify athero
        oneside_wall_pred = (pred == 2).float()[:, step * i : step * (i + 1)] 
        non_zeros, filter_mask = filtering_mask(oneside_wall_mask)
        if non_zeros.any():
            for idx in non_zeros:
                lumen_mask_slice = oneside_lumen_mask[None, ..., idx]
                lumen_pred_slice = oneside_lumen_pred[None, ..., idx]

                wall_mask_slice = oneside_wall_mask[None, ..., idx] 
                wall_pred_slice = oneside_wall_pred[None, ..., idx]
                dice_metric(wall_pred_slice, wall_mask_slice)
                recall_metric(wall_pred_slice, wall_mask_slice)
                precision_metric(wall_pred_slice, wall_mask_slice)
                hd = compute_hausdorff_distance(wall_pred_slice, wall_mask_slice, spacing=spacing)
                hd_metric.append(hd.mean())
                sd = compute_average_surface_distance(wall_pred_slice, wall_mask_slice, spacing=spacing)
                sd_metric.append(sd.mean())

                b_num = betti_number(wall_pred_slice[0, 0])
                bnum_metric.append(b_num)

                target_nwi = (wall_pred_slice.sum() - lumen_pred_slice.sum()) / wall_pred_slice.sum()
                src_nwi = (wall_mask_slice.sum() - lumen_mask_slice.sum()) / wall_mask_slice.sum()
                nwi_diff = abs(target_nwi - src_nwi) / src_nwi if src_nwi > 0 else 1.0
                nwi_metric.append(nwi_diff)
    if hd_metric:
        recall = recall_metric.compute().item()
        precision = precision_metric.compute().item()
        dice = dice_metric.aggregate().item()

        hd_metric = avoid_nan(hd_metric)    
        hd = sum(hd_metric) / len(hd_metric)
        sd_metric = avoid_nan(sd_metric)    
        sd = sum(sd_metric) / len(sd_metric)
        nwi_metric = avoid_nan(nwi_metric)
        nwi_diff = sum(nwi_metric) / len(nwi_metric)
        b_num = sum(bnum_metric) / len(bnum_metric)
        # print('final', b_num)

        print(f"({case_file}) Dice: %.3f, ASD: %.3f, HD: %.3f, Recall: %.3f, Precision: %.3f, BettiN: %.3f, NWI Diff: %.3f" % (dice, sd, hd, recall, precision, b_num, nwi_diff))
        return {"dice": dice, "asd": sd, "hd95": hd, "recall": recall, "precision": precision, 'betti': b_num, "nwi diff": nwi_diff}
    else:
        return None

if __name__ == '__main__':
    main()
