import os

import lightning.pytorch as L
import torch
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric

from networks import define_optimizer
from networks.MTNet import MTNet
from utils.tools import filtering_mask, save_segmentation_result


class Model_MTNet(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.log_dir = cfg.log_dir
        self.lambda_mri = 1.0
        self.lambda_ct = cfg.lambda_ct
        self.lambda_pseudo = cfg.lambda_pseudo
        self.netS = MTNet(in_channels=1, out_channels=2, nsf=cfg.nsf, aligner_branch=cfg.aligner_branch)
        self.criterion = DiceCELoss(include_background=False, softmax=True, to_onehot_y=True, reduction="mean")
        self.inferer = SlidingWindowInferer(roi_size=cfg.roi_size, sw_batch_size=2, overlap=0.25)
        self.val_acc_metric = DiceMetric(include_background=False)
        self.automatic_optimization = False

    def forward(self, x: torch.Tensor):
        pred = self.inferer(x, self.netS)
        return pred.argmax(1, keepdim=True)

    def training_step(self, batch, batch_idx=None):
        optimizer = self.optimizers()

        mri_loss = self.compute_mri_loss(batch, optimizer)
        self.log("loss_mri", mri_loss)

        ct_loss = self.compute_ct_loss(batch, optimizer)
        self.log("loss_ct", ct_loss)

        total_loss = mri_loss * self.lambda_mri + ct_loss * self.lambda_ct
        self.log("loss_total", total_loss, prog_bar=True)
        return total_loss

    def compute_mri_loss(self, batch, optimizer):
        mri_image = batch["mri"]["image"]
        mri_mask = batch["mri"]["mask"]
        mri_sparse = batch["mri"]["sparse"]

        optimizer.zero_grad()
        self.netS.forward_branch = 1
        mri_pred = self.netS(mri_image)

        loss_mri, valid_batch_count = 0, 0
        # Loop over each batch element
        for b in range(len(mri_image)):
            anno_id, _ = filtering_mask(mri_mask[b])
            sparse_id, _ = filtering_mask(mri_sparse[b])
            if len(anno_id) > 0:
                valid_batch_count += 1
                pseudo_id = torch.tensor([i for i in anno_id if i not in sparse_id], dtype=int)
                # Calculate the weights for the sparse and pseudo-sparse annotations
                weight_sparse = len(sparse_id) / len(anno_id)
                weight_pseudo = len(pseudo_id) / len(anno_id)

                loss_sparse, loss_pseudo = 0, 0
                if weight_sparse > 0:
                    loss_sparse = self.criterion(mri_pred[b : b + 1, ..., sparse_id], mri_mask[b : b + 1, ..., sparse_id])
                if weight_pseudo > 0:
                    loss_pseudo = self.criterion(mri_pred[b : b + 1, ..., pseudo_id], mri_mask[b : b + 1, ..., pseudo_id])
                # Calculate the batch loss as the weighted sum of the sparse and pseudo-sparse losses
                loss_batch = loss_sparse * weight_sparse + loss_pseudo * weight_pseudo * self.lambda_pseudo
                loss_mri += loss_batch
        if valid_batch_count > 0:
            loss_mri /= valid_batch_count
            self.manual_backward(loss_mri)
            optimizer.step()
        return loss_mri

    def compute_ct_loss(self, batch, optimizer):
        if self.lambda_ct <= 0:
            return 0
        ct_image = batch["ct"]["image"]
        ct_mask = batch["ct"]["mask"]

        optimizer.zero_grad()
        self.netS.forward_branch = 2
        ct_pred = self.netS(ct_image)
        loss_ct = self.criterion(ct_pred, ct_mask)
        self.manual_backward(loss_ct)
        optimizer.step()
        return loss_ct

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        mri_image = batch["mri"]["image"]
        mri_mask = batch["mri"]["mask"]
        self.netS.forward_branch = 1
        mri_pred = self.inferer(mri_image, self.netS)
        mri_pred = mri_pred.argmax(1, keepdim=True)
        self.val_acc_metric(mri_pred, mri_mask)

        ct_image = batch["ct"]["image"]
        ct_mask = batch["ct"]["mask"]
        self.netS.forward_branch = 2
        ct_pred = self.inferer(ct_image, self.netS)
        ct_pred = ct_pred.argmax(1, keepdim=True)

        if self.local_rank == 0 and batch_idx == 0:
            save_segmentation_result(
                dim=3,
                image=mri_image[0],
                masks=[mri_mask[0], mri_pred[0]],
                save_dir=os.path.join(self.log_dir, f"epoch_{self.current_epoch}_mri_vis.gif"),
            )
            save_segmentation_result(
                dim=3,
                image=ct_image[0],
                masks=[ct_mask[0], ct_pred[0]],
                save_dir=os.path.join(self.log_dir, f"epoch_{self.current_epoch}_ct_vis.gif"),
            )

    def on_validation_epoch_end(self) -> None:
        val_acc = self.val_acc_metric.aggregate()
        self.log("val_acc", val_acc)
        self.val_acc_metric.reset()

    def configure_optimizers(self):
        optimizer = define_optimizer(
            self.cfg.opt_name,
            self.netS.parameters(),
            lr=self.cfg.lr,
            **self.cfg.opt_args,
        )
        return [optimizer]
