"""Test the data.raw module"""
from typing import List

import pytest

from src.data.raw import Raw


def test_implementation():
    """Test the implementation of a Raw dataset"""

    class WrongRawImplementation(Raw):
        @classmethod
        def get_root_path(cls) -> str:
            return ""

        @classmethod
        def get_annotations_keys(cls) -> List[str]:
            return ["not_class_id", "other_key"]

        @classmethod
        def is_available(cls) -> bool:
            return True

        def get_path(self) -> str:
            return ""

        def get_name(self):
            return ""

        def _load_images(self) -> List[str]:
            return None

        def _load_annotations(self) -> List[dict]:
            return None

    with pytest.raises(NotImplementedError):
        assert WrongRawImplementation()
