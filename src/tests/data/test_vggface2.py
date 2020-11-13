"""Test the data.vggface2 module"""

import os

import pytest

from src.data.vggface2 import VGGFace2

vggface2 = VGGFace2("test")


@pytest.mark.skipif(not VGGFace2.is_available(), reason="requires the vggface2 dataset")
def test_init():
    """Test the initialization of the VGGFace2 class"""
    assert vggface2[0]
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


@pytest.mark.skipif(not VGGFace2.is_available(), reason="requires the vggface2 dataset")
def test_get_image_annotations():
    """Test the get_image_annotations function"""
    data = {"class_id": "n000001", "image_id": "0001", "face_id": "01"}

    with pytest.raises(TypeError):
        assert vggface2.get_image_annotations()

    with pytest.raises(IndexError):
        assert vggface2.get_image_annotations(index=len(vggface2._images) + 1)

    with pytest.raises(ValueError):
        assert vggface2.get_image_annotations(image_path="/")
    with pytest.raises(ValueError):
        assert vggface2.get_image_annotations(image_path="wrong_naming.jpg")

    assert vggface2.get_image_annotations(index=0) == data
    assert (
        vggface2.get_image_annotations(
            image_path=os.path.join(
                VGGFace2.get_root_path(),
                data["class_id"],
                f"{data['image_id']}_{data['face_id']}.jpg",
            )
        )
        == data
    )


@pytest.mark.skipif(not VGGFace2.is_available(), reason="requires the vggface2 dataset")
def test_get_image():
    """Test the get_image function"""
    class_id = "n000001"

    data = os.path.join(VGGFace2.get_root_path(), class_id, "0001_01.jpg")

    with pytest.raises(IndexError):
        assert vggface2.get_image(index=len(vggface2._images) + 1)

    with pytest.raises(ValueError):
        assert vggface2.get_image(class_id="wrong_class")

    assert vggface2.get_image()
    assert vggface2.get_image(index=0) == data
    assert vggface2.get_image(class_id=class_id)


@pytest.mark.skipif(not VGGFace2.is_available(), reason="requires the vggface2 dataset")
def test_get_images():
    """Test the get_images function"""
    class_id = "n000001"

    with pytest.raises(ValueError):
        assert vggface2.get_images(class_id="wrong_class")

    with pytest.raises(ValueError):
        assert vggface2.get_images(n_images=-1)

    assert len(vggface2.get_images(n_images=10)) == 10
    assert len(vggface2.get_images(indexes=[0, 1, 2])) == 3
    assert len(vggface2.get_images(class_id=class_id)) == 424
