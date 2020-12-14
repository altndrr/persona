"""Collection of utility functions about data."""
from glob import glob
from typing import List, Tuple

import torch
from facenet_pytorch import fixed_image_standardization
from torch import Tensor


def convert_classes_to_indexes(labels, classes):
    """
    Convert a list of labels representing classes to their corresponding indexes.
    More precisely, convert TripletDataset labels to the index of the class in the
    dataset, while keeping the current label for a FolderDataset dataset.

    :param labels: list of labels to convert.
    :param classes: list of all the classes in the dataset.
    """
    if all([l in classes for l in labels]):
        labels = [classes.index(label) for label in labels]
    return labels


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
