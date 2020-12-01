"""Collection of utility functions for the src.data module"""
import os
from glob import glob
from typing import List, Tuple

import numpy as np
import torch
import torchvision
from facenet_pytorch import fixed_image_standardization
from torch import Tensor
from torchvision import transforms

from src.utils import path


def get_triplet_dataloader(
    dataset: torch.utils.data.Dataset, batch_size: int, num_workers: int
) -> torch.utils.data.DataLoader:
    """
    Get the dataloader for a TripletDataset.

    :param dataset: dataset to use in the data loader
    :param batch_size: size of the batch
    :param num_workers: number of workers
    :return: data loader
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_triplet,
        num_workers=num_workers,
    )


def get_vggface2_classes(split) -> List[str]:
    """
    Get the classes of a split of the VGGFace2 dataset

    :param split: split of the dataset, either `train` or `test`
    :return: dataset classes
    """

    assert split in ["test", "train"]

    classes = [
        os.path.basename(folder)
        for folder in glob(
            os.path.join(path.get_project_root(), "data", "raw", "vggface2", split, "*")
        )
    ]

    return classes


def get_lfw_dataset() -> torch.utils.data.Dataset:
    """
    Get the dataset containing all the aligned images of LFW.

    :return: lfw dataset.
    """
    dataset_path = os.path.join(path.get_project_root(), "data", "processed", "lfw")
    trans = transforms.Compose(
        [np.float32, transforms.ToTensor(), fixed_image_standardization]
    )

    return torchvision.datasets.ImageFolder(dataset_path, transform=trans)


def get_image_dataloader(
    dataset: torchvision.datasets.ImageFolder, batch_size: int, num_workers: int
) -> torch.utils.data.DataLoader:
    """
    Get the dataloader for a ImageFolder dataset.

    :param dataset: dataset to use in the data loader
    :param batch_size: size of the batch
    :param num_workers: number of workers
    :return: data loader
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=torch.utils.data.SequentialSampler(dataset),
    )


def collate_pil(x):
    """Collate PIL images"""
    out_x, out_y = [], []
    for xx, yy in x:
        out_x.append(xx)
        out_y.append(yy)
    return out_x, out_y


def collate_triplet(
    batch: List[Tuple[List[Tensor], List[str]]]
) -> Tuple[Tensor, List[str]]:
    batch_images, batch_classes = [], []

    for images, classes in batch:
        batch_images.append(torch.stack(images))
        batch_classes.extend(classes)

    batch_images = torch.cat(batch_images)

    return batch_images, batch_classes
