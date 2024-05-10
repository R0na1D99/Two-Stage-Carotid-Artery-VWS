from .model_mtnet import Model_MTNet
from .model_wall import Model_Wall

def define_model(cfg, init=True):
    model_mapping = {
        'wall': Model_Wall,
        'mtnet_interp': Model_MTNet,
        'mtnet_sam': Model_MTNet,
    }

    if cfg.stage in model_mapping:
        if init:
            return model_mapping[cfg.stage](cfg)
        else:
            return model_mapping[cfg.stage]
    else:
        raise ValueError(f"Model stage '{cfg.stage}' not recognized.")
