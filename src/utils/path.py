"""Collection of generic functions"""
import os
from typing import List


def split(path: str) -> List[str]:
    """
    Split a path into all of its parts

    :param path: path to file or directory
    :return: parts of the path
    """
    parts = []

    head, tail = os.path.split(path)
    parts.append(tail)

    while head != "":
        head, tail = os.path.split(head)
        parts.append(tail)

    parts = parts[::-1]

    return parts


def change_data_category(path: str, category: str) -> str:
    """
    Change the path to a data folder or data item from a category (e.g. `raw`), to another (e.g.
    `processed`.

    :param path: path to data
    :param category: folder for overwrite
    :return ValueError:
    """
    parts = split(path)

    if "data" not in parts:
        raise ValueError(
            f"{path} is not a path to a folder containing data or a data item"
        )

    index = parts.index("data")
    if index + 1 >= len(parts):
        raise ValueError(f"{path} does not contain a category to substitute")

    if parts[index + 1] not in ["interim", "output", "processed", "raw"]:
        raise ValueError(f"{path} does not contain a category to substitute")

    parts[index + 1] = category

    return os.path.join(*parts)


def get_project_root() -> str:
    """Get the root of the project"""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
