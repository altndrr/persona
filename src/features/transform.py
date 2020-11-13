"""Collection of functions to transform data"""

import os
from typing import List, Union

from PIL import Image
from torch import Tensor
from torchvision.transforms import ToPILImage, ToTensor


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

        tensors.append(ToTensor()(image))

    if len(tensors) == 1:
        tensors = tensors[0]

    return tensors


def tensors_to_images(*tensors: Tensor) -> Union[Image.Image, List[Image.Image]]:
    """"
    Get a list of tensors and convert them to images

    :param tensors: tensor representation of images
    :return: images
    """
    images = [ToPILImage()(tensor) for tensor in tensors]

    if len(images) == 1:
        images = images[0]

    return images
