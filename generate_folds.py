from pathlib import Path
import os, torch
from torch.utils.data import random_split
import json

if __name__ == '__main__':
    dataset = 'COSMOS' # careII COSMOS
    root_dir = Path(os.getenv("DATA_ROOT"))
    raw_dir = root_dir / dataset / "train_data"
    input_dir = root_dir / dataset / "preprocessed" / "mri_nii_raw"
    if dataset == 'careII':
        all_cases = [case.name for case in raw_dir.glob("0_P*")]
    elif dataset == 'COSMOS':
        all_cases = [case.name for case in raw_dir.glob("[0-9]*")]
    all_cases = sorted(all_cases)
    for i in range(5):
        torch.manual_seed(42 + i)
        train_cases, val_cases, test_cases = random_split(all_cases, [30, 5, 15])
        train_cases, val_cases, test_cases = list(train_cases), list(val_cases), list(test_cases)
        data = {'train':train_cases, 'val':val_cases, 'test':test_cases}
        with open(root_dir / dataset / f'fold_{i}.json', 'w') as f:
            json.dump(data, f) 
            print('Saved', f'fold_{i}.json')