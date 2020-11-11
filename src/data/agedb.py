"""Collection of functions for the AgeDB dataset"""

import os
from glob import glob

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
