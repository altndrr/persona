"""Collection of functions for the AgeDB dataset"""

import os
from glob import glob
from random import randint

ROOT_PATH = os.path.join("data", "raw", "agedb")

ANNOTATION_KEYS = ["image_id", "class_id", "age", "gender"]
IS_AVAILABLE = len(glob(os.path.join(ROOT_PATH, "*.jpg"))) > 0


def _get_image_annotations(image_path: str) -> dict:
    """
    Given the path to an image, returns a dictionary containing its annotations

    :param image_path: path to the image
    :return: image annotations
    """

    filename, _ = os.path.splitext(os.path.basename(image_path))
    values = filename.split("_")

    annotations = dict(zip(ANNOTATION_KEYS, values))

    return annotations


IMAGES = glob(os.path.join(ROOT_PATH, "*.jpg"))
ANNOTATIONS = [_get_image_annotations(image) for image in IMAGES]


def get_image_annotations(image_id: int = None, image_path: str = None) -> dict:
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
            for annotation in ANNOTATIONS
            if annotation["image_id"] == image_id
        ]

        if len(annotations) != 1:
            raise ValueError(f"`{image_id}` is an invalid image identifier")

        annotations = annotations[0]
    elif image_path is not None:
        if image_path not in IMAGES:
            raise ValueError(
                f"`{image_path} does not refer to an image of the dataset`"
            )

        index = IMAGES.index(image_path)
        annotations = ANNOTATIONS[index]
    else:
        raise TypeError("either `image_id` or `image_path` must be passed")

    return annotations


def get_image(image_id: int = None, class_id: str = None) -> str:
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
            for i, image in enumerate(IMAGES)
            if ANNOTATIONS[i]["image_id"] == image_id
        ]

        if len(image) != 1:
            raise ValueError(f"`{image_id}` is an invalid image identifier")

        image = image[0]
    elif class_id is not None:
        class_images = [
            image
            for i, image in enumerate(IMAGES)
            if ANNOTATIONS[i]["class_id"] == class_id
        ]

        if len(class_images) == 0:
            raise ValueError(f"`{class_id}` is an invalid class identifier")

        image = class_images[randint(0, len(class_images) - 1)]
    else:
        image = IMAGES[randint(0, len(IMAGES) - 1)]

    return image


def get_images(image_ids: list = None, class_id: str = None, n_images: int = 5) -> list:
    """
    Get a list of paths to images of the dataset. If neither the list of image ids nor
    of class of images is passed, a list of n random image is returned.

    :param image_ids: list of unique identifiers of images
    :param class_id: unique identifier of a class of images
    :param n_images: number of random images to retrieve
    :return: list of path to images
    """
    if image_ids:
        images = [get_image(image_id=image_id) for image_id in image_ids]
    elif class_id:
        images = [
            image
            for i, image in enumerate(IMAGES)
            if ANNOTATIONS[i]["class_id"] == class_id
        ]

        if len(images) == 0:
            raise ValueError(f"`{class_id}` is an invalid class identifier")
    else:
        if n_images >= len(IMAGES) or n_images < 1:
            raise ValueError(f"`{n_images}` is an invalid number of images to retrieve")

        images = [get_image() for _ in range(n_images)]

    return images
