from typing import Dict

import lightning as L
import torch
from torch import nn
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from vision_transformers_auair.utils.misc import reduce_dict


class TrainModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        postprocessors: Dict[str, nn.Module],
        image_width: int = 864,
        image_height: int = 512,
        **kwargs,
    ):
        super().__init__()

        self.model = model
        self.model.train()
        self.model.requires_grad_(True)
        self.model.backbone.requires_grad_(True)
        self.epoch = kwargs.get("epoch", 0)
        self.lr_noise = kwargs.get("lr_noise", None)

        self.criterion = criterion
        self.postprocessors = postprocessors

        # mAP metric
        self.map_metric_val = MeanAveragePrecision(iou_type="bbox")
        self.map_metric_train = MeanAveragePrecision(iou_type="bbox")
        self.image_width = image_width
        self.image_height = image_height

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        samples, targets = batch["pixel_values"], batch["labels"]
        self.criterion.train()

        outputs = self.model(samples)
        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict
        )

        loss_dict_reduced = reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        map_train_result = self.map_metric_train(
            self.postprocessors["bbox"](
                outputs,
                target_sizes=torch.tensor(
                    [
                        [self.image_height, self.image_width]
                        for img in batch["pixel_values"]
                    ],
                    device=batch["pixel_values"].device,
                ),
            ),
            self.postprocess_gt(targets),
        )
        self.log("train_loss", losses_reduced_scaled, prog_bar=True, logger=True)
        self.log(
            "train_loss_box", loss_dict_reduced["loss_bbox"], prog_bar=True, logger=True
        )
        self.log(
            "train_loss_ce", loss_dict_reduced["loss_ce"], prog_bar=True, logger=True
        )
        self.log(
            "train_loss_giou",
            loss_dict_reduced["loss_giou"],
            prog_bar=True,
            logger=True,
        )
        self.log("train_map", map_train_result["map"], prog_bar=True, logger=True)

        return losses_reduced_scaled

    def validation_step(self, batch, batch_idx):
        samples, targets = batch["pixel_values"], batch["labels"]

        outputs = self.model(samples)
        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict
        )

        loss_dict_reduced = reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        self.map_metric_val.update(
            self.postprocessors["bbox"](
                outputs,
                target_sizes=torch.tensor(
                    [
                        [self.image_height, self.image_width]
                        for img in batch["pixel_values"]
                    ],
                    device=batch["pixel_values"].device,
                ),
            ),
            self.postprocess_gt(targets),
        )
        self.log("val_loss", losses_reduced_scaled, prog_bar=True, logger=True)
        self.log(
            "val_loss_box", loss_dict_reduced["loss_bbox"], prog_bar=True, logger=True
        )
        self.log(
            "val_loss_ce", loss_dict_reduced["loss_ce"], prog_bar=True, logger=True
        )
        self.log(
            "val_loss_giou", loss_dict_reduced["loss_giou"], prog_bar=True, logger=True
        )

        return losses_reduced_scaled

    def on_training_epoch_end(self):
        self.map_metric_train.reset()

    def on_validation_epoch_end(self):
        metrics = self.map_metric_val.compute()
        self.log("val_mAP", metrics["map"], prog_bar=True)
        self.map_metric_val.reset()

    def configure_optimizers(self):
        optimizer = self.build_optimizer()
        scheduler, _ = self.create_scheduler(optimizer)
        return [optimizer], [scheduler]

    def create_scheduler(self, optimizer):
        num_epochs = self.epoch
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100)
        return lr_scheduler, num_epochs

    def build_optimizer(self):
        head = []
        backbone_decay = []
        backbone_no_decay = []

        skip = set()
        if hasattr(self.model.backbone, "no_weight_decay"):
            skip = set(self.model.backbone.no_weight_decay())

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if "backbone" not in name:
                head.append(param)
            else:
                if (
                    len(param.shape) == 1
                    or name.endswith(".bias")
                    or name.split(".")[-1] in skip
                ):
                    backbone_no_decay.append(param)
                else:
                    backbone_decay.append(param)

        param_dicts = [
            {"params": head},
            {"params": backbone_no_decay, "weight_decay": 0.0, "lr": 1e-4},
            {"params": backbone_decay, "lr": 1e-4},
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=1e-4, weight_decay=1e-4)
        return optimizer

    def postprocess_gt(self, targets):
        gts = []
        for t in targets:
            boxes = t["boxes"]
            labels = t["labels"]

            if boxes.shape[0] == 0:
                gts.append(
                    {
                        "boxes": boxes.new_empty((0, 4)),
                        "labels": labels.new_empty((0,), dtype=torch.int64),
                    }
                )
                continue

            boxes_xyxy = self.box_cxcywh_to_xyxy(boxes)
            boxes_xyxy = boxes_xyxy * torch.tensor(
                [
                    self.image_width,
                    self.image_height,
                    self.image_width,
                    self.image_height,
                ],
                device=boxes.device,
                dtype=boxes.dtype,
            )

            gts.append({"boxes": boxes_xyxy, "labels": labels})
        return gts

    @staticmethod
    def box_cxcywh_to_xyxy(boxes):
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        return torch.stack([x1, y1, x2, y2], dim=-1)
