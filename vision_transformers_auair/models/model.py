from typing import Dict, List
import torch
import lightning as L
import torch.nn.functional as F
from torch import nn

from vision_transformers_auair.models.nn.configuration_yolos import YolosConfig
from vision_transformers_auair.models.nn.yolos import YolosForObjectDetection
from vision_transformers_auair.utils.losses import SetCriterion, HungarianMatcher


class YOLOS(nn.Module):
    def __init__(self, config_path: str, num_classes: int = 8, pretrained: bool = False):
        super().__init__()

        config = YolosConfig.from_pretrained(config_path)
        config.num_labels = num_classes
        config.image_size = [512, 864]

        if pretrained:
            self.model = YolosForObjectDetection(config)
            state_dict = YolosForObjectDetection.from_pretrained(config_path).state_dict()
            filtered_dict = {k: v for k, v in state_dict.items() if "class_labels_classifier" not in k}
            self.model.load_state_dict(filtered_dict, strict=False)
        else:
            self.model = YolosForObjectDetection(config)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.model(pixel_values=x, return_dict=True)
        return {
            "logits": outputs.logits,
            "pred_boxes": outputs.pred_boxes
        }


class TrainModule(L.LightningModule):
    def __init__(
        self,
        config_path: str,
        pretrained: bool = True,
        num_classes: int = 8,
        matcher_costs: Dict = None,
        loss_weights: Dict = None,
    ):
        super().__init__()

        self.model = YOLOS(config_path, num_classes, pretrained)

        matcher = HungarianMatcher(
            class_cost=matcher_costs.get("class", 1.0),
            bbox_cost=matcher_costs.get("bbox", 5.0),
            giou_cost=matcher_costs.get("giou", 2.0),
        )

        self.criterion = SetCriterion(
            num_classes=num_classes,
            matcher=matcher,
            weight_dict=loss_weights or {
                "loss_ce": 1,
                "loss_bbox": 5,
                "loss_giou": 2
            },
            eos_coef=0.1,
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images = batch["pixel_values"]
        labels = batch["labels"] 

        outputs = self(images)
        loss_dict = self.criterion(outputs, labels)

        loss = sum(loss_dict[k] * self.criterion.weight_dict[k] for k in loss_dict.keys() if k in self.criterion.weight_dict)

        self.log_dict({f"train_{k}": v.item() for k, v in loss_dict.items()}, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
        return [optimizer], [scheduler]

    
        



if __name__ == "__main__":
    # Example usage
    config_path = "hustvl/yolos-tiny"
    yolos_model = YOLOS(config_path, pretrained=True)

    # Dummy input
    dummy_input = torch.randn(
        1, 3, 512, 864
    )  # Batch size of 1, 3 channels, height and width as per YOLOS
    outputs = yolos_model(dummy_input)

    print(outputs["logits"].shape)  # Should print the shape of logits
    print(outputs["pred_boxes"].shape)  # Should print the shape of predicted boxes
