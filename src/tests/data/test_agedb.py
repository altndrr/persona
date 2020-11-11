"""Test the data.agedb module"""

import pytest

import src.data.agedb as agedb

test_annotations = [
    ({}, TypeError, None),
    ({"image_id": -1}, ValueError, None),
    ({"image_id": "-1"}, ValueError, None),
    ({"image_id": True}, ValueError, None),
    (
        {"image_id": 0},
        None,
        {"age": "35", "class_id": "MariaCallas", "gender": "f", "image_id": "0"},
    ),
    ({"image_id": "-1"}, ValueError, None),
    ({"image_path": "/"}, ValueError, None),
    ({"image_path": "wrong_naming.jpg"}, ValueError, None),
    (
        {"image_path": agedb.get_image(image_id=0)},
        None,
        {"age": "35", "class_id": "MariaCallas", "gender": "f", "image_id": "0"},
    ),
]


@pytest.mark.skipif(not agedb.IS_AVAILABLE, reason="requires the agedb dataset")
@pytest.mark.parametrize("options, error, output", test_annotations)
def test_get_annotations(options, error, output):
    """Test the get_annotation function"""
    if error:
        with pytest.raises(error):
            agedb.get_image_annotations(**options)
    elif output:
        assert agedb.get_image_annotations(**options) == output
    else:
        assert agedb.get_image_annotations(**options)


test_image = [
    ({}, None, None),
    ({"image_id": -1}, ValueError, None),
    ({"image_id": "-1"}, ValueError, None),
    ({"image_id": True}, ValueError, None),
    ({"image_id": 0}, None, None),
    ({"class_id": -1}, ValueError, None),
    ({"class_id": "-1"}, ValueError, None),
    ({"class_id": True}, ValueError, None),
    ({"class_id": "MariaCallas"}, None, None),
]


@pytest.mark.skipif(not agedb.IS_AVAILABLE, reason="requires the agedb dataset")
@pytest.mark.parametrize("options, error, output", test_image)
def test_get_image(options, error, output):
    """Test the get_image function"""
    if error:
        with pytest.raises(error):
            agedb.get_image(**options)
    elif output:
        assert agedb.get_image(**options) == output
    else:
        assert agedb.get_image(**options)


test_images = [
    ({}, None, 5),
    ({"image_ids": [-1]}, ValueError, 1),
    ({"image_ids": [True]}, ValueError, 1),
    ({"image_ids": ["-1"]}, ValueError, 1),
    ({"image_ids": [0, 1, 2]}, None, 3),
    ({"class_id": -1}, ValueError, 1),
    ({"class_id": True}, ValueError, 1),
    ({"class_id": "MariaCallas"}, None, 24),
    ({"n_images": -1}, ValueError, 1),
]


@pytest.mark.skipif(not agedb.IS_AVAILABLE, reason="requires the agedb dataset")
@pytest.mark.parametrize("options, error, output_len", test_images)
def test_get_images(options, error, output_len):
    """Test the get_images function"""
    if error:
        with pytest.raises(error):
            agedb.get_images(**options)
    else:
        assert len(agedb.get_images(**options)) == output_len
