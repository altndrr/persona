"""Test the main module."""

import os
from glob import glob

import src
from src.utils import path


def test_init():
    """Test the __init__.py file"""
    assert hasattr(src, "__version__")


def test_main():
    """Test the main file"""
    assert os.system("python -m src") == 1
    assert os.system("python -m src -h") == 0
    assert os.system("python -m src --help") == 0
    assert os.system("python -m src --version") == 0


def test_models_list():
    """Test the models list command"""
    assert os.system("python -m src models list") == 0


def test_triplets_list():
    """Test the triplets list command"""
    assert os.system("python -m src triplets list") == 0


def test_triplets_make():
    """Test the triplets make command"""
    pre_files = glob(
        os.path.join(path.get_project_root(), "data", "processed", "triplets", "*.npy")
    )
    assert os.system("python -m src triplets make") == 1
    assert os.system("python -m src triplets make X agedb") == 1
    assert os.system("python -m src triplets make 1 agedb -w Y") == 1
    assert os.system("python -m src triplets make 1 agedb -w 1") == 0

    assert os.system("python -m src triplets make 1 vggface2 --split test -w 1") == 0
    assert os.system("python -m src triplets make 1 wrong_name -w 1") == 1

    post_files = glob(
        os.path.join(path.get_project_root(), "data", "processed", "triplets", "*.npy")
    )

    created_files = list(set(post_files) - set(pre_files))
    for file in created_files:
        os.remove(file)
