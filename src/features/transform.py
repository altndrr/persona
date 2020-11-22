"""Collection of functions to transform data"""

import os
from typing import List, Union

import numpy as np
import torchvision
from facenet_pytorch import MTCNN
from PIL import Image
from torch import Tensor

mtcnn = MTCNN(
    image_size=160,
    margin=14,
    selection_method="center_weighted_size",
    post_process=False,
)


def align_images(
    *images: Union[str, Image.Image], fail_ok=True
) -> Union[Tensor, List[Tensor]]:
    """
    Take image paths and images and try to align them using the MTCNN module

    :param images: paths to images or images
    :param fail_ok: suppress TypeErrors if alignment is impossible
    :return: aligned image
    :raises FileNotFoundError, TypeError:
    """
    aligned_images = []

    for image in images:
        if isinstance(image, str):
            if not os.path.exists(image) or not os.path.isfile(image):
                raise FileNotFoundError(f"{image} is an invalid image path")

            image = Image.open(image)

        # Convert to RGB to fix the number of channels to three.
        image = image.convert("RGB")

        try:
            aligned_images.append(mtcnn(image))
        except TypeError:
            # A face couldn't be detected in the image.
            if not fail_ok:
                raise TypeError

    if len(aligned_images) == 1:
        aligned_images = aligned_images[0]

    return aligned_images


def images_to_tensors(*images: Union[str, Image.Image]) -> Union[Tensor, List[Tensor]]:
    """
    Get a list of image paths and images and convert them to tensors

    :param images: paths to images and images
    :return: tensors
    :raises FileNotFoundError:
    """
    tensors = []

    for image in images:
        if isinstance(image, str):
            if not os.path.exists(image) or not os.path.isfile(image):
                raise FileNotFoundError(f"{image} is an invalid image path")

            image = Image.open(image)

        # Convert to RGB to fix the number of channels to three.
        image = image.convert("RGB")

        transformations = torchvision.transforms.Compose(
            [np.float32, torchvision.transforms.ToTensor()]
        )
        tensors.append(transformations(image))

    if len(tensors) == 1:
        tensors = tensors[0]

    return tensors


def tensors_to_images(*tensors: Tensor) -> Union[Image.Image, List[Image.Image]]:
    """"
    Get a list of tensors and convert them to images

    :param tensors: tensor representation of images
    :return: images
    """
    trans = torchvision.transforms.Compose([torchvision.transforms.ToPILImage()])
    images = [trans(tensor) for tensor in tensors]

    if len(images) == 1:
        images = images[0]

    return images
