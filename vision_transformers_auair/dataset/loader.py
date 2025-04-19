import json
import logging
import os
from typing import Dict, Tuple

import albumentations as A
import cv2
import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

dataset_logger = logging.getLogger("dataset_logger")
dataset_logger.setLevel(logging.INFO)
np.random.seed(42)


class AUAirDataset(Dataset):
    original_image_size = (1080, 1920)

    def __init__(
        self,
        data_dir: str,
        task: str = "train",
        transform: transforms.Compose = None,
        shuffle: bool = False,
        image_height: int = 512,
        image_width: int = 864,
    ):
        self.data_dir = data_dir
        self.transform = transform
        self.image_height = image_height
        self.image_width = image_width
        self.images = []
        self.annotations = []
        self.task = task
        self.train_augmentation = self.set_train_augmentation()
        self.val_augmentation = A.Compose(
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1)
        )

        with open(os.path.join(data_dir, "annotations.json"), "r") as file:
            metadata = json.load(file)
            for annotation in metadata["annotations"]:
                self.images.append(
                    os.path.join(data_dir, "images", annotation["image_name"])
                )
                self.annotations.append(annotation["bbox"])
        self.split(task=task)

        if shuffle:
            data = list(zip(self.images, self.annotations))
            np.random.shuffle(data)
            self.images, self.annotations = zip(*data)
        self.images = list(self.images)
        self.annotations = list(self.annotations)

        # TODO: remove later
        # self.images = self.images[:10]
        # self.annotations = self.annotations[:10]

        for i in range(len(self.annotations)):
            bboxes = []
            labels = []
            for bbox in self.annotations[i]:
                xmin = bbox["left"]
                ymin = bbox["top"]
                width = bbox["width"]
                height = bbox["height"]
                if height < 5 or width < 5:
                    continue
                xmax = xmin + width
                ymax = ymin + height
                cx = (xmin + xmax) / 2 / self.original_image_size[1]
                cy = (ymin + ymax) / 2 / self.original_image_size[0]
                w = width / self.original_image_size[1]
                h = height / self.original_image_size[0]
                bboxes.append([cx, cy, w, h])
                labels.append(bbox["class"])

            self.annotations[i] = {"boxes": bboxes, "labels": labels}

        dataset_logger.info(f"Loaded {len(self.images)} images and annotations.")
        dataset_logger.info(f"Loaded {len(self.images)} images for {task} task.")
        dataset_logger.info(
            f"Loaded {len(self.annotations)} annotations for {task} task."
        )
        dataset_logger.info(f"Data directory: {data_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict]:
        image_path = self.images[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_width, self.image_height))
        annotations = self.annotations[index]
        if self.task == "train":
            augmented = self.train_augmentation(
                image=np.array(image),
                bboxes=np.array(annotations["boxes"]),
                labels=np.array(annotations["labels"]),
            )
            image, annotations["boxes"], annotations["labels"] = (
                augmented["image"],
                augmented["bboxes"],
                augmented["labels"],
            )
        else:
            image = self.val_augmentation(image=image)["image"]

        # self.visualize_bbox(image ,annotations)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        target = {
            "boxes": torch.tensor(annotations["boxes"], dtype=torch.float32),
            "labels": torch.tensor(annotations["labels"], dtype=torch.int64),
        }

        return image, target

    def set_train_augmentation(self):
        compose_list = []
        compose_list.append(
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2, p=0.3
                    ),
                    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                ]
            )
        )
        compose_list.append(
            A.RandomSizedBBoxSafeCrop(height=512, width=864, erosion_rate=0.2, p=0.5),
        )
        compose_list.append(A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=10, p=0.5))
        compose_list.append(
            A.HorizontalFlip(p=0.5),
        )

        compose_list.append(
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1)
        )

        compose = A.Compose(
            compose_list,
            p=1,
            bbox_params=A.BboxParams(
                format="yolo", label_fields=["labels"], min_visibility=0.2
            ),
        )
        return compose

    def visualize_bbox(self, image, annotations):
        """
        Visualize bounding boxes on the images.
        """

        for bbox in annotations["boxes"]:
            x1 = int((bbox[0] - bbox[2] / 2) * self.image_width)
            y1 = int((bbox[1] - bbox[3] / 2) * self.image_height)
            x2 = int((bbox[0] + bbox[2] / 2) * self.image_width)
            y2 = int((bbox[1] + bbox[3] / 2) * self.image_height)

            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv2.imwrite("bbox_visualization.jpg", image)

    def split(self, train_ratio: float = 0.8, task: str = "train"):
        if task == "train":
            self.images = self.images[: int(0.75 * len(self.images))]
            self.annotations = self.annotations[: int(0.75 * len(self.annotations))]

        elif task == "val":
            self.images = self.images[
                int(0.75 * len(self.images)) : int(0.9 * len(self.images))
            ]
            self.annotations = self.annotations[
                int(0.75 * len(self.annotations)) : int(0.9 * len(self.annotations))
            ]

        elif task == "test":
            self.images = self.images[int(0.9 * len(self.images)) :]
            self.annotations = self.annotations[int(0.9 * len(self.annotations)) :]

        else:
            raise ValueError(f"Unknown task: {task}. Use 'train', 'val', or 'test'.")


def collate_fn(batch):
    images, targets = zip(*batch)
    return {"pixel_values": torch.stack(images), "labels": list(targets)}


class AuAirDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        image_height: int = 512,
        image_width: int = 864,
        transform: transforms.Compose = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.image_height = image_height
        self.image_width = image_width

    def setup(self, stage: str):
        self.train_dataset = AUAirDataset(
            self.data_dir,
            task="train",
            transform=self.transform,
            shuffle=True,
            image_height=self.image_height,
            image_width=self.image_width,
        )
        self.val_dataset = AUAirDataset(
            self.data_dir,
            task="val",
            transform=self.transform,
            image_height=self.image_height,
            image_width=self.image_width,
        )
        self.test_dataset = AUAirDataset(
            self.data_dir,
            task="test",
            transform=self.transform,
            image_height=self.image_height,
            image_width=self.image_width,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )


if __name__ == "__main__":
    data_dir = os.path.join("vision_transformers_auair", "dataset", "auair2019")
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    dataset = AUAirDataset(data_dir, transform)

    for img, metadata in dataset:
        print(img.size(), metadata)
        break
