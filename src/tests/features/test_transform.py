"""Test the features.transform module"""

import pytest
import torch
from PIL import Image

from src.data.agedb import AgeDB
from src.features import transform

age = AgeDB()


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
