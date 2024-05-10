import os
import lightning.pytorch as pl
import torch
from networks import define_S, define_scheduler, define_optimizer
from utils.tools import save_segmentation_result
from utils.ti_loss import TI_Loss
from monai.inferers import SliceInferer
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.losses import DiceCELoss


class Model_Wall(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.lr = cfg.lr
        self.root_dir = cfg.root_dir
        self.log_dir = cfg.log_dir
        self.batch_size = cfg.batch_size

        H, W, D = cfg.mri_crop_size
        cfg.roi_size = (H // 2, W)
        self.netS = define_S(2, cfg)
        self.deep_sup = cfg.netS == 'unetplusplus'

        self.criterion_pixel = DiceCELoss(softmax=True, lambda_dice=1., lambda_ce=0.5) # compute all channels, softmax
        if cfg.use_ti:
            self.weight_extra = 1e-4 
            connectivity = 4 
            self.criterion_extra = TI_Loss(dim=2, connectivity=connectivity, inclusion=[[1, 2], [2, 0]], exclusion=[[0, 1]], min_thick=1)
        
        self.val_acc_metric = DiceMetric(include_background=False)
        
        self.inferer = SliceInferer(
            roi_size=cfg.roi_size,
            sw_batch_size=32,
            spatial_dim=2, 
            progress=False,
        )


    def forward(self, batch):
        only_last = lambda x: self.netS(x)[-1] if self.deep_sup else self.netS(x)
        pred = self.inferer(batch, only_last).argmax(1, keepdim=True)
        return pred

    def training_step(self, batch, batch_idx):
        image: torch.Tensor = batch["image"]
        mask: torch.Tensor = batch["mask"]
        gaussian = batch["gaussian"]
        inputs = torch.cat([image, gaussian], 1)
        pred = self.netS(inputs)
        
        if self.deep_sup:
            weights = [1/8, 1/4, 1/2, 1]
            loss = sum([w * self.criterion_pixel(out, mask) for w, out in zip(weights, pred)]) / sum(weights)
            pred = pred[-1]
        else:
            loss = self.criterion_pixel(pred, mask)
        
        if self.cfg.use_ti:
            loss += self.criterion_extra(pred, mask.argmax(1, keepdim=True)) * self.weight_extra
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        image: torch.Tensor = batch["image"]
        mask: torch.Tensor = batch["mask"]
        gaussian = batch["gaussian"]
        inputs = torch.cat([image, gaussian], 1)

        pred = self.netS(inputs)
        if self.deep_sup:
            pred = pred[-1]
        pred = pred.argmax(1, keepdim=True)
        self.val_acc_metric(y_pred=pred, y=mask.argmax(1, keepdim=True))
        if self.local_rank == 0 and batch_idx == 0:
            pred = AsDiscrete(to_onehot=3)(pred[0])
            save_segmentation_result(
                2, image[0], [mask[0], pred],
                save_dir=os.path.join(self.log_dir, f"epoch_{self.current_epoch}_vis.gif"),
            )

    def on_validation_epoch_end(self) -> None:
        val_acc = self.val_acc_metric.aggregate()
        self.log("val_acc", val_acc)
        self.val_acc_metric.reset()

    def configure_optimizers(self):
        optimizer = define_optimizer(self.cfg.opt_name, self.netS.parameters(), lr=self.cfg.lr, **self.cfg.opt_args)
        if self.cfg.lr_scheduler != 'none':
            scheduler = define_scheduler(optimizer, self.cfg.lr_scheduler, self.cfg.max_epochs)
            return [optimizer], [scheduler]
        else:
            return optimizer
