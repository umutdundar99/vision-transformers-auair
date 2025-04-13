import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops import generalized_box_iou


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


class HungarianMatcher(nn.Module):
    def __init__(self, class_cost=1, bbox_cost=5, giou_cost=2):
        super().__init__()
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost

    def forward(self, outputs, targets):
        with torch.no_grad():
            bs, num_queries = outputs["logits"].shape[:2]
            out_prob = outputs["logits"].softmax(-1)
            out_bbox = outputs["pred_boxes"]

            indices = []
            for b in range(bs):
                tgt_ids = targets[b]["class_labels"]
                tgt_bbox = targets[b]["boxes"]

                cost_class = -out_prob[b][:, tgt_ids]

                cost_bbox = torch.cdist(out_bbox[b], tgt_bbox, p=1)

                cost_giou = -generalized_box_iou(
                    box_cxcywh_to_xyxy(out_bbox[b]),
                    box_cxcywh_to_xyxy(tgt_bbox)
                )

                C = self.class_cost * cost_class + \
                    self.bbox_cost * cost_bbox + \
                    self.giou_cost * cost_giou
                indices.append(linear_sum_assignment(C.cpu()))

            return [(torch.as_tensor(i, dtype=torch.int64),
                     torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef

        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes):
        src_logits = outputs["logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([
            t["class_labels"][J] for t, (_, J) in zip(targets, indices)
        ])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes,
            dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2), target_classes, self.empty_weight
        )
        return {"loss_ce": loss_ce}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)
        ))

        losses = {
            "loss_bbox": loss_bbox.sum() / num_boxes,
            "loss_giou": loss_giou.sum() / num_boxes,
        }
        return losses

    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        num_boxes = sum(len(t["boxes"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_boxes = torch.clamp(num_boxes, min=1).item()

        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices, num_boxes))
        losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))

        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([
            torch.full_like(src, i) for i, (src, _) in enumerate(indices)
        ])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
