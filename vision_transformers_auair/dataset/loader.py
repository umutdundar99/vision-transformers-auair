import json
import logging
import os
from typing import Dict, Tuple

import lightning as L
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

dataset_logger = logging.getLogger("dataset_logger")
dataset_logger.setLevel(logging.INFO)


class AUAirDataset(Dataset):
    def __init__(
        self, data_dir: str, task: str = "train", transform: transforms.Compose = None
    ):
        self.data_dir = data_dir
        self.transform = transform

        self.images = []
        self.annotations = []
        self.task = task

        with open(os.path.join(data_dir, "annotations.json"), "r") as file:
            metadata = json.load(file)
            for annotation in metadata["annotations"]:
                self.images.append(
                    os.path.join(data_dir, "images", annotation["image_name"])
                )
                self.annotations.append(annotation["bbox"])

        dataset_logger.info(f"Loaded {len(self.images)} images and annotations.")
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
        dataset_logger.info(f"Loaded {len(self.images)} images for {task} task.")
        dataset_logger.info(
            f"Loaded {len(self.annotations)} annotations for {task} task."
        )
        dataset_logger.info(f"Data directory: {data_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict]:
        image_path = self.images[index]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        metadata = {
            "bbox": self.annotations[index],
            "image_name": os.path.basename(image_path),
        }

        return image, metadata


class AuAirLoader(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        transform: transforms.Compose = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform

    def setup(self, stage: str):
        self.train_dataset = AUAirDataset(
            self.data_dir, task="train", transform=self.transform
        )
        self.val_dataset = AUAirDataset(
            self.data_dir, task="val", transform=self.transform
        )
        self.test_dataset = AUAirDataset(
            self.data_dir, task="test", transform=self.transform
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
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
