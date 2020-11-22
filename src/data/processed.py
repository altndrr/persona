"""Collection of classes for processed datasets."""
import os
from glob import glob
from typing import List, Tuple

import numpy as np
import torch.utils.data
import torchvision.transforms
from facenet_pytorch import fixed_image_standardization
from torch import Tensor

from src.features import transform


class TripletDataset(torch.utils.data.Dataset):
    """Dataset composed of triplets."""

    def __init__(self, triplet_file_id: int, transforms: bool = False):
        """
        Instantiate a TripletDataset.

        :param triplet_file_id: id of the file of triplets
        """
        super().__init__()
        self._filepath = self._get_filepath(triplet_file_id)
        if self._filepath is None:
            raise ValueError(f"{triplet_file_id} is an invalid file id")

        self._triplets = np.load(self._filepath)

        transformations = [fixed_image_standardization]

        if transforms:
            transformations.append(torchvision.transforms.RandomHorizontalFlip(0.5))

        self._transforms = torchvision.transforms.Compose(transformations)

    def __getitem__(self, idx: int) -> Tuple[List[Tensor], List[str]]:
        images, classes = self._triplets[idx][:3], self._triplets[idx][3:]
        images = transform.images_to_tensors(*images)

        images = [self._transforms(image) for image in images]

        return images, classes

    def __len__(self):
        return len(self._triplets)

    def get_triplet(self, idx: int) -> Tuple[List[str], List[str]]:
        """
        Get paths and classes of a triplet instead of tensors and classes.

        :param idx: id of the triplet.
        """
        paths, classes = self._triplets[idx][:3], self._triplets[idx][3:]

        return paths, classes

    @staticmethod
    def _get_filepath(triplet_file_id: int) -> str:
        """
        Retrieve the path to the triplet file given its id

        :param triplet_file_id:  id of the file of triplets
        :return: path to triplet file
        """
        triplet_path = os.path.join("data", "processed", "triplets")
        triplet_files = glob(os.path.join(triplet_path, "*.npy"))

        triplet_file_id = str(triplet_file_id).zfill(2)

        path = None
        for file in triplet_files:
            if os.path.basename(file).startswith(triplet_file_id):
                path = file
                break

        return path

    def get_name(self) -> str:
        """
        Extract the name of the dataset from the name of the triplet file on which is based.
        """
        basename = os.path.basename(self._filepath)
        basename, _ = os.path.splitext(basename)
        parts = basename.split("_")[1:-1]
        return "_".join(parts)

    @staticmethod
    def collate_fn(
        batch: List[Tuple[List[Tensor], List[str]]]
    ) -> Tuple[Tensor, List[str]]:
        """Collate function for the dataset

        :param batch: batch of data to collate
        :return:
        """

        batch_images, batch_classes = [], []

        for images, classes in batch:
            batch_images.append(torch.stack(images))
            batch_classes.extend(classes)

        batch_images = torch.cat(batch_images)

        return batch_images, batch_classes
