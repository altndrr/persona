"""Collection of classes and functions for the AgeDB dataset"""

import os
from glob import glob
from random import randint


class AgeDB:
    """AgeDB main class"""

    ROOT_PATH = os.path.join("data", "raw", "agedb")

    ANNOTATION_KEYS = ["image_id", "class_id", "age", "gender"]
    IS_AVAILABLE = os.path.exists(ROOT_PATH)

    def __init__(self):
        """Init function of the class"""

        self._path = self.ROOT_PATH
        self._images = self._load_images()
        self._annotations = self._load_annotations()

    def _load_annotations(self) -> list:
        """
        Load the list of annotations

        :return: image annotations
        """

        annotations = []

        for image in self._images:
            filename, _ = os.path.splitext(os.path.basename(image))
            values = filename.split("_")

            annotations.append(dict(zip(self.ANNOTATION_KEYS, values)))

        return annotations

    def _load_images(self) -> list:
        """
        Load the list of images

        :return: image paths
        """
        target = os.path.join(self.ROOT_PATH, "*.jpg")
        images = glob(target)

        return images

    def get_image_annotations(
        self, image_id: int = None, image_path: str = None
    ) -> dict:
        """
        Given the path to an image, returns a dictionary containing its annotations

        :param image_id: unique identifier of the image
        :param image_path: path to the image
        :return: image annotations
        :raises ValueError:
        """

        if image_id is not None:
            image_id = str(image_id)
            annotations = [
                annotation
                for annotation in self._annotations
                if annotation["image_id"] == image_id
            ]

            if len(annotations) != 1:
                raise ValueError(f"`{image_id}` is an invalid image identifier")

            annotations = annotations[0]
        elif image_path is not None:
            if image_path not in self._images:
                raise ValueError(
                    f"`{image_path} does not refer to an image of the dataset`"
                )

            index = self._images.index(image_path)
            annotations = self._annotations[index]
        else:
            raise TypeError("either `image_id` or `image_path` must be passed")

        return annotations

    def get_image(self, image_id: int = None, class_id: str = None) -> str:
        """
        Get the path to an image of the dataset. If neither the id of the image nor the id
        of a class of images is passed, a random image is returned.

        :param image_id: unique identifier of an image
        :param class_id: unique identifier of a class of images
        :return: path to the image
        :raises ValueError:
        """

        if image_id is not None:
            image_id = str(image_id)
            image = [
                image
                for i, image in enumerate(self._images)
                if self._annotations[i]["image_id"] == image_id
            ]

            if len(image) != 1:
                raise ValueError(f"`{image_id}` is an invalid image identifier")

            image = image[0]
        elif class_id is not None:
            class_images = [
                image
                for i, image in enumerate(self._images)
                if self._annotations[i]["class_id"] == class_id
            ]

            if len(class_images) == 0:
                raise ValueError(f"`{class_id}` is an invalid class identifier")

            image = class_images[randint(0, len(class_images) - 1)]
        else:
            image = self._images[randint(0, len(self._images) - 1)]

        return image

    def get_images(
        self, image_ids: list = None, class_id: str = None, n_images: int = 5
    ) -> list:
        """
        Get a list of paths to images of the dataset. If neither the list of image ids nor
        of class of images is passed, a list of n random image is returned.

        :param image_ids: list of unique identifiers of images
        :param class_id: unique identifier of a class of images
        :param n_images: number of random images to retrieve
        :return: list of path to images
        """
        if image_ids:
            images = [self.get_image(image_id=image_id) for image_id in image_ids]
        elif class_id:
            images = [
                image
                for i, image in enumerate(self._images)
                if self._annotations[i]["class_id"] == class_id
            ]

            if len(images) == 0:
                raise ValueError(f"`{class_id}` is an invalid class identifier")
        else:
            if n_images >= len(self._images) or n_images < 1:
                raise ValueError(
                    f"`{n_images}` is an invalid number of images to retrieve"
                )

            images = [self.get_image() for _ in range(n_images)]

        return images
