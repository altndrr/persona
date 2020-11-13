"""Test the features.transform module"""

import pytest
import torch
from PIL import Image
from torchvision.transforms import ToPILImage

from src.data.agedb import AgeDB
from src.features import transform

age = AgeDB()


@pytest.mark.skipif(not AgeDB.is_available(), reason="requires the agedb dataset")
def test_align_images():
    """Test the align_images function"""
    image_path_1 = age.get_image(index=0)
    image_path_2 = age.get_image(index=1)

    generated_image = ToPILImage()(torch.ones(3, 160, 160))

    with pytest.raises(FileNotFoundError):
        transform.align_images("")

    with pytest.raises(TypeError):
        transform.align_images(generated_image, fail_ok=False)

    assert len(transform.align_images(generated_image, fail_ok=True)) == 0
    assert isinstance(transform.align_images(image_path_1), torch.Tensor)
    assert isinstance(transform.align_images(image_path_1, image_path_2), list)
    assert len(transform.align_images(image_path_1, image_path_2)) == 2
    assert isinstance(transform.align_images(), list)
    assert len(transform.align_images()) == 0


@pytest.mark.skipif(not AgeDB.is_available(), reason="requires the agedb dataset")
def test_conversions():
    """Test the images_to_tensors and tensors_to_images functions"""
    image_path_1 = age.get_image(index=0)
    image_path_2 = age.get_image(index=1)

    with pytest.raises(FileNotFoundError):
        transform.images_to_tensors("")

    assert isinstance(transform.images_to_tensors(image_path_1), torch.Tensor)
    output_1 = transform.images_to_tensors(image_path_1, image_path_2)
    assert isinstance(output_1, list)
    assert len(output_1) == 2

    assert isinstance(transform.tensors_to_images(output_1[0]), Image.Image)
    assert isinstance(transform.tensors_to_images(*output_1), list)
