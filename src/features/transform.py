"""Collection of functions to transform data"""

import os
from typing import List, Union

import numpy as np
import torchvision
from facenet_pytorch import MTCNN
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader

from src.utils import path
from src.utils.data import collate_pil

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


def align_dataset(data_dir, batch_size, workers, device):
    """
    Given the directory of a dataset, tries to align all of the images inside it.

    The function returns the list of image batches in which an image couldn't be
    aligned (a single failed alignment stops the rest of the batch).
    To align an higher number of images, it is suggested to align one-by-one the returned
    list (i.e. with a batch_size of 1).

    :param data_dir: path to the data directory
    :param batch_size: dimension of a batch
    :param workers: number of workers to use
    :param device: device to use for detection and alignment
    :return: list of images batches that couldn't be completely aligned
    """
    mtcnn_ = MTCNN(
        image_size=160,
        margin=14,
        selection_method="center_weighted_size",
        post_process=False,
        device=device,
    )

    class SquarePad:
        def __call__(self, image):
            w, h = image.size
            max_wh = np.max([w, h])
            hp = int((max_wh - w) / 2)
            vp = int((max_wh - h) / 2)
            padding = (hp, vp, hp, vp)
            return torchvision.transforms.functional.pad(image, padding)

    trans = torchvision.transforms.Compose(
        [SquarePad(), torchvision.transforms.Resize((255, 255)),]
    )

    dataset = torchvision.datasets.ImageFolder(data_dir, transform=trans)
    dataset.samples = [(p, p) for p, _ in dataset.samples]

    loader = DataLoader(
        dataset, num_workers=workers, batch_size=batch_size, collate_fn=collate_pil
    )

    unable_to_align = []
    for i, (x, paths) in enumerate(loader):
        crops = [path.change_data_category(p, "processed") for p in paths]
        try:
            mtcnn_(x, save_path=crops)
        except:
            unable_to_align.extend(crops)
        print("\rBatch {} of {}".format(i + 1, len(loader)), end="")

    return unable_to_align


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
