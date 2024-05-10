import argparse
import os
import warnings

from lightning.pytorch import Trainer, callbacks, loggers, strategies
from monai.utils import get_torch_version_tuple
from monai.utils.misc import set_determinism

from datasets import define_dataset
from models import define_model
from utils.tools import init_configs, load_configs, save_current_configs

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="mtnet", type=str)
    parser.add_argument("-r", "--resume", default=None, type=str)
    parser.add_argument("-m", "--model_name", default="test", type=str)
    parser.add_argument("-f", "--fold", default=0, type=int)
    args = parser.parse_args()
    if args.resume:
        cfg = load_configs(resume=args.resume)
    else:
        cfg = init_configs("configs", args)
        cfg = save_current_configs(cfg=cfg)

    cfg.resume = os.path.join(args.resume, "last.ckpt") if args.resume else None
    if cfg.use_amp and get_torch_version_tuple() < (1, 6):
        raise RuntimeError("AMP feature only exists in PyTorch version greater than v1.6.")

    train(cfg)


def train(cfg):
    set_determinism(42)
    pl_model = define_model(cfg)
    dataset = define_dataset(cfg)

    checkpoint_callback = callbacks.ModelCheckpoint(
        dirpath=cfg.log_dir,
        filename="{epoch:02d}-{val_acc:.4f}",
        monitor="val_acc",
        mode="max",
        save_top_k=5,
        save_last=True,
    )

    lr_monitor = callbacks.LearningRateMonitor(logging_interval="epoch")

    model_callbacks = [checkpoint_callback, lr_monitor]

    tb_logger = loggers.TensorBoardLogger(cfg.log_dir, default_hp_metric=False)

    extra_params = {}

    if cfg.use_amp:
        extra_params["precision"] = "16-mixed"

    if len(cfg.gpus) > 0:
        extra_params["accelerator"] = "gpu"
        extra_params["devices"] = cfg.gpus
        if len(cfg.gpus) > 1:
            extra_params["strategy"] = strategies.DDPStrategy(find_unused_parameters=False)
    else:
        extra_params["accelerator"] = "cpu"
        print("WARNING: CPU only, this will be slow!")

    trainer = Trainer(
        max_epochs=cfg.max_epochs,
        default_root_dir=cfg.log_dir,
        callbacks=model_callbacks,
        logger=tb_logger,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=1,
        log_every_n_steps=10,
        **extra_params
    )

    trainer.fit(model=pl_model, datamodule=dataset, ckpt_path=cfg.resume)


if __name__ == "__main__":
    main()
