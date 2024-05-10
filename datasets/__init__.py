from .dataset_mtnet import MTNet_Dataset
from .dataset_wall import Wall_Dataset
from types import SimpleNamespace

def define_dataset(cfg: SimpleNamespace):
    """
    Define the dataset based on the stage specified in the configuration.

    Args:
        cfg (SimpleNamespace): A configuration object that contains the stage name.

    Returns:
        Dataset: A dataset object of the specified stage.

    Raises:
        ValueError: If the specified stage is not recognized.

    """
    dataset_mapping = {
        'wall': Wall_Dataset,
        'mtnet_interp': MTNet_Dataset,
        'mtnet_sam': MTNet_Dataset,
    }

    if cfg.stage in dataset_mapping:
        return dataset_mapping[cfg.stage](cfg)
    else:
        raise ValueError(f"Dataset stage '{cfg.stage}' not recognized.")
