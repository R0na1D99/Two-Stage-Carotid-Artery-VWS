# Carotid Vessel Wall Segmentation Through Domain Aligner, Topological Learning, and Segment Anything for Sparse Annotation in MRI Images



This is the official PyTorch implementation of the paper:\
[Carotid Vessel Wall Segmentation Through Domain Aligner, Topological Learning, and Segment Anything for Sparse Annotation in MRI Images](https://google.com)\
X. Li, X. Ouyang, J. Zhang, Z. Ding, Y. Zhang, Z. Xue, F. Shi, and D. Shen

## Installation
To install the required dependencies, run the following command in your terminal:

```shell
pip install -r requirements.txt.
```


## Dataset Preparation
1. Download the [COSMOS2022](https://vessel-wall-segmentation-2022.grand-challenge.org/) dataset and extract the files. Since our CTA data is restricted, you will need to use your own external dataset.

2. Organize your data in the following structure:
    ```
    - DATA_ROOT
        - COSMOS
            - train_data
                - 3
                - 4
                - ...
        - CTA_dataset
            - image
                - **.nii.gz
                ...
            - mask
                - **.nii.gz
                ...
    ```

4. Create a `invalid.txt` file within the COSMOS folder with the following content. The images are divided down the center to create two halves, and there are no annotations on the specific half sides of these images:
```
28_R_image.nii.gz
29_R_image.nii.gz
42_L_image.nii.gz
43_R_image.nii.gz
47_R_image.nii.gz
52_R_image.nii.gz
7_L_image.nii.gz
```

5. For external validation on CARE-II, organize the data similarly and copy the test data into the train data folder for cross-validation. The content for `invalid.txt` for external validation would be:

```
0_P176_U_R_image.nii.gz
0_P204_U_L_image.nii.gz
0_P204_U_R_image.nii.gz
0_P252_U_L_image.nii.gz
0_P448_U_R_image.nii.gz
0_P460_U_L_image.nii.gz
0_P759_U_R_image.nii.gz
0_P955_U_R_image.nii.gz
```

6. Set the DATA_ROOT environment variable as follows:

```shell
export DATA_ROOT=/data/
```

7. Run the [to_nii](to_nii.ipynb)  to convert DICOM files to NII format and generate interpolated annotations if needed for external validation.

8. Follow the [sam_annotation](sam_annotation.ipynb) notebook to install SAM and generate SAM interpolated masks. The files will be automatically saved under `$DATA_ROOT/COSMOS(or careII)/preprocessed/mri_nii_raw`. Use tools such as ITK-SNAP to manually examine the generated masks and eliminate clearly failed slices.

9. Run the [generate_folds](generate_folds.py) to generate corresponding json files for each fold.

## Training
Configure the training parameters for your training stage in the [configs](configs) folder.

Then run `train.py` with your configuration file, like this:
```shell
python train.py --config mtnet
```

## Testing
Execute the [eval_lumen.ipynb](eval_lumen.ipynb) notebook to evaluate lumen segmentation using MT-Net. Additionally, run [eval_wall.ipynb](eval_wall.ipynb) to assess the vessel wall segmentation using our two-stage framework.


### Acknowledgement

This repository is built using [MONAI](MONAI) and [Lightning](lightning).

### License
This project is released under the Apache license. Please see the [LICENSE](License) file for more information.

### Citation
If our work is useful for your research, please consider citing:
```
```
