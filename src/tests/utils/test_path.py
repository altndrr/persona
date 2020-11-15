"""Test the utils.path module"""

import os

import pytest

from src.utils import path


def test_split():
    """Test the split function"""
    assert path.split("") == [""]
    assert path.split("data\\raw") == ["data", "raw"]


def test_change_data_category():
    """Test the change_data_category function"""
    with pytest.raises(ValueError):
        path.change_data_category("", "raw")
    with pytest.raises(ValueError):
        path.change_data_category("utils", "raw")
    with pytest.raises(ValueError):
        path.change_data_category("root\\data", "raw")
    with pytest.raises(ValueError):
        path.change_data_category("root\\data\\wrong_category", "raw")

    assert (
        path.change_data_category("data\\raw\\dataset", "processed")
        == "data\\processed\\dataset"
    )
    assert path.change_data_category("data\\raw", "interim") == "data\\interim"


def test_get_project_path():
    """Test the get_project_path function"""
    assert os.path.basename(path.get_project_root()) == "face-recognition"
