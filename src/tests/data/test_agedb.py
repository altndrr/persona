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
