"""Collection of classes and functions for the VGGFace2 dataset"""

import os
from random import randint


class VGGFace2:
    """VGGFace2 main class"""

    ROOT_PATH = os.path.join("data", "raw", "vggface2")

    ANNOTATION_KEYS = ["class_id", "image_id", "face_id"]
    IS_AVAILABLE = os.path.exists(ROOT_PATH)

    def __init__(self, split: str):
        """
        Init function of the class

        :param split: split of the dataset, value must be either `train` or `test`
        """

        assert split in ["train", "test"]

        self._split = split
        self._path = os.path.join(self.ROOT_PATH, self._split)
        self._images = self._load_images()
        self._annotations = self._load_annotations()

    def _load_annotations(self) -> list:
        """
        Load the list of annotations of the defined split

        :return: image annotations
        """

        annotation_name = f"{self._split}_annotations.txt"
        annotation_path = os.path.join(self.ROOT_PATH, annotation_name)

        if os.path.exists(annotation_path):
            all_values = [line.strip() for line in open(annotation_path).readlines()]
            annotations = [
                dict(zip(self.ANNOTATION_KEYS, values.split(", ")))
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
                annotations.append(dict(zip(self.ANNOTATION_KEYS, values)))

            open(annotation_path, "w").writelines(all_values)

        return annotations

    def _load_images(self) -> list:
        """
        Load the list of the images of the specified split

        :return: image paths
        """
        images = [
            os.path.join(self.ROOT_PATH, *os.path.split(line.strip()))
            for line in open(
                os.path.join(self.ROOT_PATH, f"{self._split}_list.txt")
            ).readlines()
        ]

        return images

    def get_image_annotations(self, index: int = None, image_path: str = None) -> dict:
        """
        Get the annotations of an image

        :param index: index of the image
        :param image_path: path to the image
        :return: image annotations
        :raises IndexError, TypeError, ValueError:
        """
        if index is not None:
            if index <= len(self._annotations) - 1:
                annotations = self._annotations[index]
            else:
                raise IndexError(f"`{index} is not a valid index`")
        elif image_path is not None:
            if image_path not in self._images:
                raise ValueError(
                    f"`{image_path} does not refer to an image of the dataset`"
                )

            index = self._images.index(image_path)
            annotations = self._annotations[index]
        else:
            raise TypeError("either `index` or `image_path` must be passed")

        return annotations

    def get_image(self, index: int = None, class_id: str = None) -> str:
        """
        Get the path to an image. If neither the id of the image nor the id of a class of images
        is passed, a random image is returned.

        :param index: index of the image
        :param class_id: unique identifier of a class of images
        :return: path to the image
        :raises IndexError, ValueError:
        """

        if index is not None:
            if index <= len(self._images) - 1:
                image = self._images[index]
            else:
                raise IndexError(f"`{index} is not a valid index`")
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
        self, indexes: list = None, class_id: str = None, n_images: int = 5
    ) -> list:
        """
        Get a list of paths to images. If neither the list of image ids nor a class of images is
        passed, a list of n random image is returned.

        :param indexes: list of indexes to images
        :param class_id: unique identifier of a class of images
        :param n_images: number of random images to retrieve
        :return: list of path to images
        :raises ValueError:
        """
        if indexes:
            images = [self.get_image(index=index) for index in indexes]
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
