"""Collection of classes and functions for the AgeDB dataset"""

import os
from glob import glob

from src.data.raw import Raw


class AgeDB(Raw):
    """AgeDB main class"""

    @classmethod
    def get_root_path(cls) -> str:
        return os.path.join("data", "raw", "agedb")

    @classmethod
    def get_annotations_keys(cls) -> list:
        return ["image_id", "class_id", "age", "gender"]

    @classmethod
    def is_available(cls) -> bool:
        return os.path.exists(cls.get_root_path())

    def get_path(self) -> str:
        return self.get_root_path()

    def _load_annotations(self) -> list:
        """
        Load the list of annotations

        :return: image annotations
        """

        annotations = []

        for image in self._images:
            filename, _ = os.path.splitext(os.path.basename(image))
            values = filename.split("_")

            annotations.append(dict(zip(self.get_annotations_keys(), values)))

        return annotations

    def _load_images(self) -> list:
        """
        Load the list of images

        :return: image paths
        """
        target = os.path.join(self.get_root_path(), "*.jpg")
        images = glob(target)

        return images
