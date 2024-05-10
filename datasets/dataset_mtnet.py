import random

from lightning import LightningDataModule
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from monai.transforms import (
    Compose,
    EnsureTyped,
    LabelFilterd,
    LoadImaged,
    SpatialPadd,
    RandAdjustContrastd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandCropByPosNegLabeld,
)
from pathlib import Path
from utils.data import get_ct_nii, get_mri_nii
from utils.tools import get_dataloader, filter_invalid, load_split    


class MTNet_Dataset(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.root_dir = Path(cfg.root_dir)

        # setup cta directories
        self.ct_input = self.root_dir / "CTA_dataset"
        self.ct_output = self.root_dir / "CTA_dataset" / "preprocessed" / "ct_nii_crop"
        self.ct_cases = [case.name[:-7] for case in (self.ct_input / "image").glob("*.nii.gz")]  # remove `.nii.gz`

        assert cfg.ct_portion <=1 and cfg.ct_portion > 0, ValueError("Wrong CTA portion! Must be (0-1]!") # for portion=0, simply run a resunet
        self.ct_cases = random.sample(self.ct_cases, int(len(self.ct_cases) * cfg.ct_portion))

        self.mri_input = self.root_dir / cfg.dataset / "preprocessed" / "mri_nii_raw"
        self.mri_output = self.root_dir / cfg.dataset / "preprocessed" / "mri_nii_crop"
        self.mri_train, self.mri_val, self.mri_test = load_split(self.root_dir / cfg.dataset, cfg.fold)

    def prepare_data(self) -> None:
        """Prepare the data for training and validation.

        This method generates the preprocessed data by calling `get_mri_nii` and `get_ct_nii`
        functions for each case in the `self.mri_train`, `self.mri_val`, and `self.ct_cases` lists.

        Notes
        -----
        This method only runs on the first GPU in distributed training so that
        the preprocessing is performed only once.
        """
        for case in self.mri_train + self.mri_val:
            get_mri_nii(self.mri_input, case, self.mri_output, self.cfg.mri_crop_size)

        for case in self.ct_cases:
            get_ct_nii(self.ct_input, case, self.ct_output, self.cfg.ct_crop_size)

    def setup(self, stage=None):
        """Prepare the data for a specific stage.

        This method gets the file list for each stage and prepares the data for training or validation.
        It calls `get_mri_nii` and `get_ct_nii` functions to obtain the file dictionaries for each case.
        It also filters out invalid files and splits the data into training and validation sets.

        Notes
        -----
        This method is run on each GPU in distributed training. It simply gets the file list
        and prepares the data for training or validation. The actual data loading and preprocessing
        is performed by the `get_mri_nii` and `get_ct_nii` functions.
        """
        print("Val cases:", *list(self.mri_val))  # examine splits are the same across platforms
        self.mri_train_files, self.mri_val_files = [], []
        for case in self.mri_train + self.mri_val:
            file_dicts = get_mri_nii(self.mri_input, case, self.mri_output, self.cfg.mri_crop_size)
            if self.cfg.stage == "mtnet_interp":
                file_dicts = [{"image": i["image"], "sparse": i["mask"], "mask": i["interp"]} for i in file_dicts]
            elif self.cfg.stage == "mtnet_sam":
                file_dicts = [{"image": i["image"], "sparse": i["mask"], "mask": i["sam"]} for i in file_dicts]
            else:
                raise KeyError(self.cfg.stage + " is not a valid stage!")

            if case in self.mri_train:
                self.mri_train_files.extend(file_dicts)
            elif case in self.mri_val:
                self.mri_val_files.extend(file_dicts)

        self.mri_train_files = filter_invalid(self.root_dir / self.cfg.dataset, self.mri_train_files)
        self.mri_val_files = filter_invalid(self.root_dir / self.cfg.dataset, self.mri_val_files)

        self.ct_train_files = []
        for case in self.ct_cases:
            file_dicts = get_ct_nii(self.ct_input, case, self.ct_output, self.cfg.ct_crop_size)
            self.ct_train_files.extend(file_dicts)

        self.ct_val_files = random.sample(
            self.ct_train_files, len(self.mri_val_files)
        )  # for visualization in validation

    def val_transforms(self):
        trans = Compose(
            [
                LoadImaged(
                    ["image", "mask", "sparse"], image_only=True, ensure_channel_first=True, allow_missing_keys=True
                ),
                SpatialPadd(["image", "mask", "sparse"], self.cfg.roi_size, allow_missing_keys=True),
                LabelFilterd(["mask", "sparse"], [1], allow_missing_keys=True),  # filter only vessel lumen
                EnsureTyped(["image", "mask", "sparse"], track_meta=False, allow_missing_keys=True),
            ]
        )
        return trans

    def train_transforms(self):
        aug_p = self.cfg.aug_p
        trans = Compose(  # `allow_missing_keys=True` for ct dict doesnt contain `sparse`
            [
                LoadImaged(
                    ["image", "mask", "sparse"], image_only=True, ensure_channel_first=True, allow_missing_keys=True
                ),
                SpatialPadd(["image", "mask", "sparse"], self.cfg.roi_size, allow_missing_keys=True),
                LabelFilterd(["mask", "sparse"], [1], allow_missing_keys=True),  # filter only vessel lumen
                RandCropByPosNegLabeld(
                    keys=["image", "mask", "sparse"],
                    label_key="mask",
                    spatial_size=self.cfg.roi_size,
                    pos=4,
                    neg=1,
                    num_samples=self.cfg.num_samples,
                    allow_missing_keys=True,
                ),
                RandGaussianNoised(["image"], prob=aug_p, std=0.01),
                RandGaussianSmoothd(
                    ["image"], prob=aug_p, sigma_x=(0.5, 1.15), sigma_y=(0.5, 1.15), sigma_z=(0.5, 1.15)
                ),
                RandScaleIntensityd(["image"], prob=aug_p, factors=0.3),
                RandAdjustContrastd(["image"], prob=aug_p, gamma=(0.8, 1.2)),
                EnsureTyped(["image", "mask", "sparse"], track_meta=False, allow_missing_keys=True),
            ]
        )
        return trans

    def train_dataloader(self):
        mri_loader = get_dataloader(self.mri_train_files, trans=self.train_transforms(), cfg=self.cfg, shuffle=True)
        ct_loader = get_dataloader(self.ct_train_files, trans=self.train_transforms(), cfg=self.cfg, shuffle=True)
        return CombinedLoader(
            {"mri": mri_loader, "ct": ct_loader}, mode="max_size_cycle"
        )  # ensure each mri file comes with a cta file

    def val_dataloader(self):
        mri_loader = get_dataloader(self.mri_val_files, trans=self.val_transforms(), cfg=self.cfg, shuffle=True)
        ct_loader = get_dataloader(self.ct_val_files, trans=self.val_transforms(), cfg=self.cfg, shuffle=True)
        return CombinedLoader(
            {"mri": mri_loader, "ct": ct_loader}, mode="max_size_cycle"
        )  # ensure each mri file comes with a cta file
