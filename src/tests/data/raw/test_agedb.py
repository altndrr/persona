"""Test the data.agedb module"""

import os

import pytest

from src.data.raw import AgeDB

agedb = AgeDB()


@pytest.mark.skipif(not AgeDB.is_available(), reason="requires the agedb dataset")
def test_init():
    """Test the initialization of the AgeDB class"""
    assert agedb[0]
    assert len(agedb.get_name()) > 0
    assert len(agedb) > 0
    assert AgeDB.is_available()
    assert len(agedb.get_path()) > 0


@pytest.mark.skipif(not AgeDB.is_available(), reason="requires the agedb dataset")
def test_get_annotations():
    """Test the get_image_annotations function"""
    class_id = "MariaCallas"
    data = {"age": "35", "class_id": "MariaCallas", "gender": "f", "image_id": "0"}
    image_path = os.path.join(AgeDB.get_root_path(), "0_MariaCallas_35_f.jpg")

    with pytest.raises(IndexError):
        assert agedb.get_annotations(len(agedb) + 1)

    assert agedb.get_annotations(0)
    assert agedb.get_annotations(0, 1)
    with pytest.raises(ValueError):
        assert agedb.get_annotations(image_paths="/")
    with pytest.raises(ValueError):
        assert agedb.get_annotations(image_paths="wrong_naming.jpg")
    assert agedb.get_annotations(image_paths=image_path)
    assert agedb.get_annotations(image_paths=[image_path, image_path])
    assert agedb.get_annotations(class_id=class_id)
    with pytest.raises(ValueError):
        assert agedb.get_annotations(class_id="wrong_class")
    assert len(agedb.get_annotations(n_random=5)) == 5
    assert agedb.get_annotations(0) == data


@pytest.mark.skipif(not AgeDB.is_available(), reason="requires the agedb dataset")
def test_get_images():
    """Test the get_images function"""
    class_id = "MariaCallas"
    data = "data\\raw\\agedb\\0_MariaCallas_35_f.jpg"

    with pytest.raises(IndexError):
        assert agedb.get_images(len(agedb) + 1)

    assert agedb.get_images(0)
    assert agedb.get_images(0, 1)
    assert agedb.get_images(class_id=class_id)
    with pytest.raises(ValueError):
        assert agedb.get_images(class_id="wrong_class")
    assert len(agedb.get_images(n_random=5)) == 5
    assert agedb.get_images(0) == data
