"""Collection of classes for raw datasets"""

import os
from abc import ABC, abstractmethod
from glob import glob
from random import randint
from typing import List, Union


class Raw(ABC):
    """Abstract base class for dataset"""

    def __init__(self):
        if "class_id" not in self.get_annotations_keys():
            raise NotImplementedError(
                "the key `class_id` must be included in the annotation keys of the class"
            )

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
    def get_name(self):
        """Get the name of the dataset"""

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


class AgeDB(Raw):
    """AgeDB main class"""

    @classmethod
    def get_root_path(cls) -> str:
        return os.path.join("data", "raw", "agedb")

    @classmethod
    def get_annotations_keys(cls) -> List[str]:
        return ["image_id", "class_id", "age", "gender"]

    @classmethod
    def is_available(cls) -> bool:
        return os.path.exists(cls.get_root_path())

    def get_name(self):
        return "agedb"

    def get_path(self) -> str:
        return self.get_root_path()

    def _load_annotations(self) -> List[dict]:
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


class VGGFace2(Raw):
    """VGGFace2 main class"""

    def __init__(self, split: str):
        """
        Init function of the class

        :param split: split of the dataset, value must be either `train` or `test`
        """
        assert split in ["train", "test"]

        self._split = split

        super().__init__()

    @classmethod
    def get_root_path(cls) -> str:
        return os.path.join("data", "raw", "vggface2")

    @classmethod
    def get_annotations_keys(cls) -> List[str]:
        return ["class_id", "image_id", "face_id"]

    @classmethod
    def is_available(cls) -> bool:
        return os.path.exists(cls.get_root_path())

    def get_name(self):
        return f"vggface2_{self._split}"

    def get_path(self) -> str:
        return os.path.join(self.get_root_path(), self._split)

    def _load_annotations(self) -> List[dict]:
        """
        Load the list of annotations of the defined split

        :return: image annotations
        """
        annotations = []

        if len(self._images) > 0:
            annotation_name = f"{self._split}_annotations.txt"
            annotation_path = os.path.join(self.get_root_path(), annotation_name)

            if os.path.exists(annotation_path):
                all_values = [
                    line.strip() for line in open(annotation_path).readlines()
                ]
                annotations = [
                    dict(zip(self.get_annotations_keys(), values.split(", ")))
                    for values in all_values
                ]
            else:
                print(
                    f"Generating (this is a one-time process) the file {annotation_name}..."
                )
                all_values = []
                annotations = []
                for image_path in self._images:
                    path, filename = os.path.split(image_path)
                    path, class_id = os.path.split(path)
                    filename, _ = os.path.splitext(filename)
                    image_id, face_id = filename.split("_")
                    values = [class_id, image_id, face_id]

                    all_values.append(", ".join(values) + "\n")
                    annotations.append(dict(zip(self.get_annotations_keys(), values)))

                open(annotation_path, "w").writelines(all_values)

        return annotations

    def _load_images(self) -> list:
        """
        Load the list of the images of the specified split

        :return: image paths
        """
        images = []
        filename = os.path.join(self.get_root_path(), f"{self._split}_list.txt")
        if os.path.exists(filename):
            images = [
                os.path.join(self.get_path(), *os.path.split(line.strip()))
                for line in open(filename).readlines()
            ]

        return images
