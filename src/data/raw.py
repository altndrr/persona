"""Abstract class for the raw datasets"""

from abc import ABC, abstractmethod
from random import randint
from typing import List, Union


class Raw(ABC):
    """Abstract base class for dataset"""

    def __init__(self):
        self._images = self._load_images()
        self._annotations = self._load_annotations()

    def __len__(self):
        return len(self._images)

    def __getitem__(self, index: int) -> (str, dict):
        return self._images[index], self._annotations[index]

    @classmethod
    @abstractmethod
    def get_root_path(cls) -> str:
        """Get the root path to the dataset"""

    @classmethod
    @abstractmethod
    def get_annotations_keys(cls) -> List[str]:
        """Get the list of keys of the annotations"""

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Check if dataset is available"""

    @abstractmethod
    def get_path(self) -> str:
        """Get the complete path to the dataset"""

    @abstractmethod
    def _load_images(self) -> List[str]:
        """Load the list of images"""

    @abstractmethod
    def _load_annotations(self) -> List[dict]:
        """Load the list of annotations"""

    def get_annotations(
        self,
        *ids: int,
        image_paths: Union[str, List[str]] = None,
        class_id=None,
        n_random: int = 1,
    ) -> Union[dict, List[dict]]:
        """
        Get image annotations. Annotations can be retrieved in three modalities: given their ids,
        specifying their class or at random.

        :param ids: list of unique identifier of images
        :param image_paths: list of paths to images
        :param class_id: unique identifier of the class of images
        :param n_random: number of annotations to get randomly
        :return: path to images
        :raises IndexError, ValueError:
        """
        annotations = []

        if ids:
            for _id in ids:
                if _id <= len(self._annotations) - 1:
                    annotations.append(self._annotations[_id])
                else:
                    raise IndexError(f"`{_id} is not a valid index`")
        elif class_id:
            annotations = [
                annotation
                for i, annotation in enumerate(self._annotations)
                if self._annotations[i]["class_id"] == class_id
            ]

            if len(annotations) == 0:
                raise ValueError(f"`{class_id}` is an invalid class identifier")
        elif image_paths:
            if isinstance(image_paths, str):
                image_paths = [image_paths]

            ids = []
            for image_path in image_paths:
                if image_path in self._images:
                    ids.append(self._images.index(image_path))
                else:
                    raise ValueError(f"{image_path} is an invalid image path")

            annotations = self.get_annotations(*ids)
        else:
            annotations = [
                self._annotations[randint(0, len(self._annotations) - 1)]
                for _ in range(n_random)
            ]

        if len(annotations) == 1:
            annotations = annotations[0]

        return annotations

    def get_images(
        self, *ids: int, class_id: str = None, n_random: int = 1
    ) -> Union[str, list]:
        """
        Get paths to images. Paths can be retrieved in three modalities: given their ids, specifying
        their class or at random.

        :param ids: list of unique identifier of images
        :param class_id: unique identifier of the class of images
        :param n_random: number of paths to get randomly
        :return: path to images
        :raises IndexError, ValueError:
        """
        images = []

        if ids:
            for _id in ids:
                if _id <= len(self._images) - 1:
                    images.append(self._images[_id])
                else:
                    raise IndexError(f"`{_id} is not a valid index`")
        elif class_id:
            images = [
                image
                for i, image in enumerate(self._images)
                if self._annotations[i]["class_id"] == class_id
            ]

            if len(images) == 0:
                raise ValueError(f"`{class_id}` is an invalid class identifier")
        else:
            images = [
                self._images[randint(0, len(self._images) - 1)] for _ in range(n_random)
            ]

        if len(images) == 1:
            images = images[0]

        return images
