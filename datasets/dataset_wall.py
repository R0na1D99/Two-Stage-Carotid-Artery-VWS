from monai.transforms import (
    Compose,
    EnsureTyped,
    RandZoomd,
    RandGaussianNoised,
    Lambdad,
    LoadImaged,
    ScaleIntensityd,
    GaussianSmoothd,
    RandAdjustContrastd,
    RandGaussianSmoothd,
    Identityd,
    RandScaleIntensityd,
)
from pathlib import Path
from lightning import LightningDataModule
from utils.tools import get_dataloader, load_split
from utils.data import get_mri_pngs


class Wall_Dataset(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.root_dir = Path(cfg.root_dir)

        self.input_dir = self.root_dir / cfg.dataset / 'preprocessed' / 'mri_nii_raw'
        self.output_dir = self.root_dir / cfg.dataset / 'preprocessed' / 'mri_png_crop'
        self.mri_train, self.mri_val, self.mri_test = load_split(self.root_dir / cfg.dataset, cfg.fold)


    def prepare_data(self) -> None:
        """Prepare the data for training and validation.

        This method generates the preprocessed data by calling `get_mri_pngs` function 
        for each case in the `self.mri_train` and `self.mri_val` lists.

        Notes
        -----
        This method only runs on the first GPU in distributed training so that
        the preprocessing is performed only once.
        """
        for case in self.mri_train + self.mri_val:
            get_mri_pngs(self.input_dir, case, self.output_dir, self.cfg.mri_crop_size)

    def setup(self, stage=None):
        """Prepare the data for a specific stage.

        This method gets the file list for each stage and prepares the data for training or validation.
        It calls `get_mri_pngs` function to obtain the file dictionaries for each case.
        It also filters out invalid files and splits the data into training and validation sets.

        Parameters
        ----------
        stage : str, optional
            The current stage of training. Can be one of "train" or "val". Default is None.

        Notes
        -----
        This method is run on each GPU in distributed training. It simply gets the file list
        and prepares the data for training or validation. The actual data loading and preprocessing
        is performed by the `get_mri_pngs` function.

        """
        self.train_files, self.val_files = [], []
        for case in self.mri_train + self.mri_val:
            file_dicts = get_mri_pngs(self.input_dir, case, self.output_dir, self.cfg.mri_crop_size)
            for file in file_dicts:
                file['gaussian'] = file['mask']
            if case in self.mri_train:
                self.train_files.extend(file_dicts)
            elif case in self.mri_val:
                self.val_files.extend(file_dicts)

    def val_transforms(self):
        # with gaussian
        trans = Compose(
            [
                LoadImaged(["image", "mask", "gaussian"], image_only=True, ensure_channel_first=True),
                ScaleIntensityd(["image", "mask", "gaussian"]),
                Lambdad(["mask"], lambda x: x.round()),
                Lambdad(["gaussian"], lambda x: x[1:2]),
                GaussianSmoothd(['gaussian'], self.cfg.sigma) if self.cfg.sigma > 0 else Identityd(['gaussian']),
                EnsureTyped(["image", "mask", "gaussian"], track_meta=False),
            ]
        )
        return trans

    def train_transforms(self):
        aug_p = self.cfg.aug_p
        trans = Compose(
            [
                LoadImaged(["image", "mask", "gaussian"], image_only=True, ensure_channel_first=True),
                ScaleIntensityd(["image", "mask", "gaussian"]),
                Lambdad(["mask"], lambda x: x.round()),
                Lambdad(["gaussian"], lambda x: x[1:2]),
                GaussianSmoothd(['gaussian'], self.cfg.sigma) if self.cfg.sigma > 0 else Identityd(['gaussian']),
                # augmentation
                RandZoomd(['gaussian'], prob=aug_p, min_zoom=0.9, max_zoom=1.1),
                RandGaussianNoised(["image"], prob=aug_p, std=0.01),
                RandGaussianSmoothd(["image"], prob=aug_p, sigma_x=(0.5, 1.15), sigma_y=(0.5, 1.15), sigma_z=(0.5, 1.15)),
                RandScaleIntensityd(["image"], prob=aug_p, factors=0.3),
                RandAdjustContrastd(["image"], prob=aug_p, gamma=(0.8, 1.2)),
                EnsureTyped(["image", "mask", "gaussian"], track_meta=False),
            ]
        )
        return trans

    def train_dataloader(self):
        loader = get_dataloader(self.train_files, trans=self.train_transforms(), cfg=self.cfg, shuffle=True)
        return loader

    def val_dataloader(self):
        loader = get_dataloader(self.val_files, trans=self.val_transforms(), cfg=self.cfg, shuffle=False)
        return loader
