import argparse
import os
import random

import cv2
import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import WandbLogger
from torchvision.ops import box_convert

import vision_transformers_auair.utils.misc as utils
from vision_transformers_auair.dataset.loader import AuAirDataModule
from vision_transformers_auair.models import build_model as build_yolos_model
from vision_transformers_auair.models.training_module import TrainModule


def get_args_parser():
    parser = argparse.ArgumentParser("Set YOLOS", add_help=False)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr_backbone", default=1e-5, type=float)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=150, type=int)
    parser.add_argument("--eval_size", default=800, type=int)

    parser.add_argument(
        "--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm"
    )

    parser.add_argument(
        "--use_checkpoint",
        action="store_true",
        help="use checkpoint.checkpoint to save mem",
    )
    parser.add_argument(
        "--sched",
        default="warmupcos",
        type=str,
        metavar="SCHEDULER",
        help='LR scheduler (default: "step", options:"step", "warmupcos"',
    )
    parser.add_argument("--lr_drop", default=100, type=int)

    parser.add_argument(
        "--warmup-lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="warmup learning rate (default: 1e-6)",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-7,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0 (1e-5)",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=0,
        metavar="N",
        help="epochs to warmup LR, if scheduler supports",
    )
    parser.add_argument(
        "--decay-rate",
        "--dr",
        type=float,
        default=0.1,
        metavar="RATE",
        help="LR decay rate (default: 0.1)",
    )

    parser.add_argument(
        "--det_token_num",
        default=25,
        type=int,
        help="Number of det token in the deit backbone",
    )
    parser.add_argument(
        "--backbone_name",
        default="tiny",
        type=str,
        help="Name of the deit backbone to use",
    )
    parser.add_argument(
        "--pre_trained",
        default="",
        help="set imagenet pretrained model path if not train yolos from scatch",
    )
    parser.add_argument(
        "--init_pe_size", nargs="+", type=int, help="init pe size (h,w)"
    )
    parser.add_argument("--mid_pe_size", nargs="+", type=int, help="mid pe size (h,w)")
    parser.add_argument(
        "--set_cost_class",
        default=1,
        type=float,
        help="Class coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_bbox",
        default=5,
        type=float,
        help="L1 box coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_giou",
        default=2,
        type=float,
        help="giou box coefficient in the matching cost",
    )

    parser.add_argument("--dice_loss_coef", default=1, type=float)
    parser.add_argument("--bbox_loss_coef", default=5, type=float)
    parser.add_argument("--giou_loss_coef", default=2, type=float)
    parser.add_argument(
        "--eos_coef",
        default=0.1,
        type=float,
        help="Relative classification weight of the no-object class",
    )

    parser.add_argument("--dataset_file", default="coco")
    parser.add_argument("--coco_path", type=str)
    parser.add_argument("--coco_panoptic_path", type=str)
    parser.add_argument("--remove_difficult", action="store_true")

    parser.add_argument(
        "--output_dir", default="", help="path where to save, empty for no saving"
    )
    parser.add_argument(
        "--device", default="cpu", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--num_workers", default=2, type=int)

    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    return parser


def main(args):
    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    args.epochs = 1000

    model, criterion, postprocessors = build_yolos_model(args)
    model.to(device)
    # HARD CODED, DO NOT CHANGE
    image_height, image_width = 512, 864
    train_module = TrainModule(
        model,
        criterion,
        postprocessors,
        image_height=image_height,
        image_width=image_width,
        **{
            "epoch": args.epochs,
            "lr_noise": None,
        },
    )
    data_module = AuAirDataModule(
        data_dir="vision_transformers_auair/dataset/auair2019",
        batch_size=2,
        num_workers=1,
        transform=None,
        image_height=image_height,
        image_width=image_width,
    )
    callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_last=True,
            save_top_k=3,
            filename="best-{epoch:02d}-{loss:.2f}",
        ),
        LearningRateMonitor(logging_interval="step"),
        RichProgressBar(),
    ]
    offline = True
    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=args.epochs,
        callbacks=callbacks,
        log_every_n_steps=1,
        precision=32,
        logger=WandbLogger(
            #entity="umudundarr-metu-middle-east-technical-university",
            project="auair",
            name="yolos-tiny-final-sample_dt25-harder-augmentation-continued",
            log_model="all" if not offline else False,
            offline=offline,
        ),
        gradient_clip_val=args.clip_max_norm,
        accumulate_grad_batches=4,
    )

    trainer.fit(train_module, data_module)
    trainer.test(train_module, data_module, ckpt_path="last")
    
    save_predictions_as_images(
        model, data_module.test_dataloader(), 0, "test", device=device
    )
    


def save_predictions_as_images(
    model, dataloader, epoch: int, save_dir: str, device="cpu"
):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    model.to(device)

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            images = batch["pixel_values"].to(device)
            outputs = model(images)
            logits = outputs["pred_logits"].softmax(-1)[..., :-1]
            scores, labels = logits.max(-1)
            boxes = box_convert(outputs["pred_boxes"], in_fmt="cxcywh", out_fmt="xyxy")
            boxes = boxes * torch.tensor(
                [images.shape[3], images.shape[2], images.shape[3], images.shape[2]],
                device=boxes.device,
            )

            for i in range(images.size(0)):
                image = images[i].cpu().permute(1, 2, 0).numpy()
                image = (image * 255).astype(np.uint8).copy()

                for box, score, label in zip(boxes[i], scores[i], labels[i]):
                    if score < 0.3:
                        continue
                    x1, y1, x2, y2 = box.int().tolist()
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        image,
                        f"Cls: {label.item()} {score.item():.2f}",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )

                save_path = os.path.join(save_dir, f"sample_{idx}_{i}.jpg")
                cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

           


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train YOLOS", parents=[get_args_parser()])
    # HARD CODED, DO NOT CHANGE
    args = parser.parse_args(
        [
            "--init_pe_size",
            "512",
            "864",
        ]
    )

    main(args)
