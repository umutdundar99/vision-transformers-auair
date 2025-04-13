from typing import Dict

import torch
from torch import nn

from vision_transformers_auair.models.nn.configuration_yolos import YolosConfig
from vision_transformers_auair.models.nn.yolos import YolosForObjectDetection


class YOLOS(nn.Module):
    def __init__(self, config_path: str, pretrained: bool = False):
        super().__init__()

        config = YolosConfig.from_pretrained(config_path)
        config.num_labels = 8
        # config.image_size = [512, 864]

        if pretrained:
            self.model = YolosForObjectDetection(config)
            state_dict = YolosForObjectDetection.from_pretrained(
                config_path
            ).state_dict()
            filtered_dict = {
                k: v
                for k, v in state_dict.items()
                if "class_labels_classifier" not in k
            }
            self.model.load_state_dict(filtered_dict, strict=False)
        else:
            self.model = YolosForObjectDetection(config)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Run inference through the YOLOS model.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Dict with:
                - logits: (B, num_queries, num_classes+1)
                - pred_boxes: (B, num_queries, 4)
        """
        outputs = self.model(pixel_values=x, return_dict=True)
        return {"logits": outputs.logits, "pred_boxes": outputs.pred_boxes}


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
