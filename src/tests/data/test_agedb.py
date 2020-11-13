"""Test the data.agedb module"""

import os

import pytest

from src.data.agedb import AgeDB

agedb = AgeDB()


@pytest.mark.skipif(not AgeDB.is_available(), reason="requires the agedb dataset")
def test_init():
    """Test the initialization of the AgeDB class"""
    assert agedb[0]
    assert AgeDB.is_available()
    assert len(agedb.get_path()) > 0


@pytest.mark.skipif(not AgeDB.is_available(), reason="requires the agedb dataset")
def test_get_image_annotations():
    """Test the get_image_annotations function"""
    data = {"age": "35", "class_id": "MariaCallas", "gender": "f", "image_id": "0"}

    with pytest.raises(TypeError):
        assert agedb.get_image_annotations()

    with pytest.raises(IndexError):
        assert agedb.get_image_annotations(index=len(agedb._images) + 1)

    with pytest.raises(ValueError):
        assert agedb.get_image_annotations(image_path="/")
    with pytest.raises(ValueError):
        assert agedb.get_image_annotations(image_path="wrong_naming.jpg")

    assert agedb.get_image_annotations(index=0) == data
    assert (
        agedb.get_image_annotations(
            image_path=os.path.join(AgeDB.get_root_path(), "0_MariaCallas_35_f.jpg")
        )
        == data
    )


@pytest.mark.skipif(not AgeDB.is_available(), reason="requires the agedb dataset")
def test_get_image():
    """Test the get_image function"""
    class_id = "MariaCallas"
    data = os.path.join(AgeDB.get_root_path(), "0_MariaCallas_35_f.jpg")

    with pytest.raises(IndexError):
        assert agedb.get_image(index=len(agedb._images) + 1)

    with pytest.raises(ValueError):
        assert agedb.get_image(class_id="wrong_class")

    assert agedb.get_image()
    assert agedb.get_image(index=0) == data
    assert agedb.get_image(class_id=class_id)


@pytest.mark.skipif(not AgeDB.is_available(), reason="requires the agedb dataset")
def test_get_images():
    """Test the get_images function"""
    class_id = "MariaCallas"

    with pytest.raises(ValueError):
        assert agedb.get_images(class_id="wrong_class")

    with pytest.raises(ValueError):
        assert agedb.get_images(n_images=-1)

    assert len(agedb.get_images(n_images=10)) == 10
    assert len(agedb.get_images(indexes=[0, 1, 2])) == 3
    assert len(agedb.get_images(class_id=class_id)) == 24
