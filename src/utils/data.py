"""Collection of utility functions for the src.data module"""
import os
from glob import glob
from typing import List

import torch

from src.data.processed import TripletDataset
from src.utils import path


def get_triplet_dataloader(
    dataset: TripletDataset, batch_size: int, num_workers: int
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
        collate_fn=TripletDataset.collate_fn,
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
