"""Test the data.agedb.vggface2 module"""

import os

import pytest

from src.data.vggface2 import VGGFace2


@pytest.mark.skipif(not VGGFace2.IS_AVAILABLE, reason="requires the vggface2 dataset")
def test_init():
    """Test the initialization of the VGGFace2 class"""

    # Remove the test_annotations.txt if present to test generation and retrieval.
    annotation_path = os.path.join(VGGFace2.ROOT_PATH, "test_annotations.txt")
    if os.path.exists(annotation_path):
        os.remove(annotation_path)

    assert VGGFace2("test")  # First call to test the generation process.
    assert VGGFace2("test")  # Second call to test the retrieval process.
    with pytest.raises(AssertionError):
        assert VGGFace2("wrong_split")


@pytest.mark.skipif(not VGGFace2.IS_AVAILABLE, reason="requires the vggface2 dataset")
def test_getter():
    """Test the getter of the VGGFace2 class"""
    test_set = VGGFace2("test")

    with pytest.raises(IndexError):
        assert test_set.get_image(len(test_set._images) + 1)
    with pytest.raises(IndexError):
        assert test_set.get_image_annotations(len(test_set._annotations) + 1)

    with pytest.raises(TypeError):
        assert test_set.get_image("-1")
    with pytest.raises(TypeError):
        assert test_set.get_image_annotations("-1")

    assert test_set.get_image(0)
    assert test_set.get_image_annotations(0)
