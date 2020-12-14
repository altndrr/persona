"""Collection of utility functions about datasets."""

import os
from glob import glob
from typing import List

import torch
from torchvision import transforms

from src.data import processed
from src.utils import data, path


def get_dataloader(
    dataset: torch.utils.data.Dataset, batch_size: int, num_workers: int,
):
    """
    Get the dataloader for a specific dataset. It supports either TripletDataset or
    FolderDataset.

    :param dataset: dataset to use in the data loader
    :param batch_size: size of the batch
    :param num_workers: number of workers
    :return: data loader
    """
    if isinstance(dataset, processed.FolderDataset):
        dataloader = get_folder_dataloader(dataset, batch_size, num_workers)
    elif isinstance(dataset, processed.TripletDataset):
        dataloader = get_triplet_dataloader(dataset, batch_size, num_workers)
    else:
        raise ValueError("The dataset passed has an unsupported type.")
    return dataloader


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
        collate_fn=data.collate_triplet,
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


def get_lfw_dataset() -> processed.FolderDataset:
    """
    Get the dataset containing all the aligned images of LFW.

    :return: lfw dataset.
    """
    dataset_path = os.path.join(path.get_project_root(), "data", "processed", "lfw")

    return processed.FolderDataset(dataset_path)


def get_vggface2_dataset(shuffle=True) -> torch.utils.data.Dataset:
    """
    Get the dataset containing all the aligned images of VGGFace2.

    :return: vggface2 dataset.
    """
    dataset_path = os.path.join(
        path.get_project_root(), "data", "processed", "vggface2", "train"
    )

    return processed.FolderDataset(dataset_path, transform=True, shuffle=shuffle)


def get_folder_dataloader(
    dataset: processed.FolderDataset, batch_size: int, num_workers: int
) -> torch.utils.data.DataLoader:
    """
    Get the dataloader for a FolderDataset.

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
