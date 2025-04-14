import argparse
import random

import lightning as L
import numpy as np
import torch
from lightning.pytorch.loggers import WandbLogger

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
    # scheduler
    # Learning rate schedule parameters
    parser.add_argument(
        "--sched",
        default="warmupcos",
        type=str,
        metavar="SCHEDULER",
        help='LR scheduler (default: "step", options:"step", "warmupcos"',
    )
    ## step
    parser.add_argument("--lr_drop", default=100, type=int)
    ## warmupcosine

    # parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
    #                     help='learning rate noise on/off epoch percentages')
    # parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
    #                     help='learning rate noise limit percent (default: 0.67)')
    # parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
    #                     help='learning rate noise std-dev (default: 1.0)')
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

    # * model setting
    parser.add_argument(
        "--det_token_num",
        default=100,
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
    # * Matcher
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
    # * Loss coefficients

    parser.add_argument("--dice_loss_coef", default=1, type=float)
    parser.add_argument("--bbox_loss_coef", default=5, type=float)
    parser.add_argument("--giou_loss_coef", default=2, type=float)
    parser.add_argument(
        "--eos_coef",
        default=0.1,
        type=float,
        help="Relative classification weight of the no-object class",
    )

    # dataset parameters
    parser.add_argument("--dataset_file", default="coco")
    parser.add_argument("--coco_path", type=str)
    parser.add_argument("--coco_panoptic_path", type=str)
    parser.add_argument("--remove_difficult", action="store_true")

    parser.add_argument(
        "--output_dir", default="", help="path where to save, empty for no saving"
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--num_workers", default=2, type=int)

    # distributed training parameters
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

    model, criterion, postprocessors = build_yolos_model(args)
    model.to(device)

    image_height, image_width = 512, 864
    module = TrainModule(
        model,
        criterion,
        postprocessors,
        image_height=image_height,
        image_width=image_width,
        kwargs={
            "epoch": 50,
            "lr_noise": None,
        },
    )

    data_module = AuAirDataModule(
        data_dir="vision_transformers_auair/dataset/auair2019",
        batch_size=10,
        num_workers=12,
        transform=None,
        image_height=image_height,
        image_width=image_width,
    )
    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=50,
        log_every_n_steps=1,
        precision=16,
        logger=WandbLogger(
            project="auair",
            name="yolos-tiny",
            offline=False,
        ),
        accumulate_grad_batches=6,
    )
    trainer.fit(module, data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train YOLOS", parents=[get_args_parser()])
    args = parser.parse_args(
        [
            "--init_pe_size",
            "512",
            "864",
        ]
    )

    main(args)
