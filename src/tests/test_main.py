"""Test the main module."""

import os

import pytest

import src

test_options = [
    (None, 1),
    ("-h", 0),
    ("--help", 0),
    ("--version", 0),
]


def test_init():
    """Test the __init__.py file"""
    assert hasattr(src, "__version__")


@pytest.mark.parametrize("options, output", test_options)
def test_main(options, output):
    """Test the main file"""
    assert os.system(f"python -m src {options}") == output
