"""Test the data.vggface2 module"""

import os

import pytest

from src.data.vggface2 import VGGFace2

vggface2 = VGGFace2("test")


@pytest.mark.skipif(not VGGFace2.is_available(), reason="requires the vggface2 dataset")
def test_init():
    """Test the initialization of the VGGFace2 class"""
    assert vggface2[0]
    assert len(vggface2) > 0
    assert VGGFace2.is_available()
    assert len(vggface2.get_path()) > 0

    # Remove the test_annotations.txt if present to test generation and retrieval.
    annotation_path = os.path.join(VGGFace2.get_root_path(), "test_annotations.txt")
    if os.path.exists(annotation_path):
        os.remove(annotation_path)

    assert VGGFace2("test")  # First call to test the generation process.
    assert VGGFace2("test")  # Second call to test the retrieval process.
    with pytest.raises(AssertionError):
        assert VGGFace2("wrong_split")


@pytest.mark.skipif(not VGGFace2.is_available(), reason="requires the agedb dataset")
def test_get_annotations():
    """Test the get_image_annotations function"""
    class_id = "n000001"
    data = {"class_id": "n000001", "image_id": "0001", "face_id": "01"}
    image_path = os.path.join(
        vggface2.get_path(),
        data["class_id"],
        f"{data['image_id']}_{data['face_id']}.jpg",
    )

    with pytest.raises(IndexError):
        assert vggface2.get_annotations(len(vggface2._images) + 1)

    assert vggface2.get_annotations(0)
    assert vggface2.get_annotations(0, 1)
    with pytest.raises(ValueError):
        assert vggface2.get_annotations(image_paths="/")
    with pytest.raises(ValueError):
        assert vggface2.get_annotations(image_paths="wrong_naming.jpg")
    assert vggface2.get_annotations(image_paths=image_path)
    assert vggface2.get_annotations(image_paths=[image_path, image_path])
    assert vggface2.get_annotations(class_id=class_id)
    with pytest.raises(ValueError):
        assert vggface2.get_annotations(class_id="wrong_class")
    assert len(vggface2.get_annotations(n_random=5)) == 5
    assert vggface2.get_annotations(0) == data


@pytest.mark.skipif(not VGGFace2.is_available(), reason="requires the agedb dataset")
def test_get_images():
    """Test the get_images function"""
    class_id = "n000001"
    data = "data\\raw\\vggface2\\test\\n000001\\0001_01.jpg"

    with pytest.raises(IndexError):
        assert vggface2.get_images(len(vggface2._images) + 1)

    assert vggface2.get_images(0)
    assert vggface2.get_images(0, 1)
    assert vggface2.get_images(class_id=class_id)
    with pytest.raises(ValueError):
        assert vggface2.get_images(class_id="wrong_class")
    assert len(vggface2.get_images(n_random=5)) == 5
    assert vggface2.get_images(0) == data
