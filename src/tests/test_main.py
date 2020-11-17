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


def test_triplets_make():
    """Test the triplets make command"""
    pre_files = glob(
        os.path.join(path.get_project_root(), "data", "processed", "triplets", "*.npy")
    )
    assert os.system("python -m src triplets make") == 1
    assert os.system("python -m src triplets make -n X -p 1 agedb") == 1
    assert os.system("python -m src triplets make -n 1 -p Y agedb") == 1

    assert os.system("python -m src triplets make -n -100 -p 1 agedb") == 1
    assert os.system("python -m src triplets make -n 1 -p -100 agedb") == 1

    assert os.system("python -m src triplets make -n 1 -p 1 agedb") == 0
    assert os.system("python -m src triplets make -n 1 -p 1 vggface2 test") == 0
    assert os.system("python -m src triplets make -n 1 -p 0 wrong_name") == 1

    post_files = glob(
        os.path.join(path.get_project_root(), "data", "processed", "triplets", "*.npy")
    )

    created_files = list(set(post_files) - set(pre_files))
    for file in created_files:
        os.remove(file)


def test_triplets_list():
    """Test the triplets list command"""
    assert os.system("python -m src triplets list") == 0
