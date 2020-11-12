"""Collection of classes and functions for the VGGFace2 dataset"""

import os

from src.data.raw import Raw


class VGGFace2(Raw):
    """VGGFace2 main class"""

    def __init__(self, split: str):
        """
        Init function of the class

        :param split: split of the dataset, value must be either `train` or `test`
        """
        assert split in ["train", "test"]

        self._split = split

        super().__init__()

    @classmethod
    def get_root_path(cls) -> str:
        return os.path.join("data", "raw", "vggface2")

    @classmethod
    def get_annotations_keys(cls) -> list:
        return ["class_id", "image_id", "face_id"]

    @classmethod
    def is_available(cls) -> bool:
        return os.path.exists(cls.get_root_path())

    def get_path(self) -> str:
        return os.path.join(self.get_root_path(), self._split)

    def _load_annotations(self) -> list:
        """
        Load the list of annotations of the defined split

        :return: image annotations
        """
        annotations = []

        if len(self._images) > 0:
            annotation_name = f"{self._split}_annotations.txt"
            annotation_path = os.path.join(self.get_root_path(), annotation_name)

            if os.path.exists(annotation_path):
                all_values = [
                    line.strip() for line in open(annotation_path).readlines()
                ]
                annotations = [
                    dict(zip(self.get_annotations_keys(), values.split(", ")))
                    for values in all_values
                ]
            else:
                print(
                    f"Generating (this is a one-time process) the file {annotation_name}..."
                )
                all_values = []
                annotations = []
                for image_path in self._images:
                    path, filename = os.path.split(image_path)
                    path, class_id = os.path.split(path)
                    filename, _ = os.path.splitext(filename)
                    image_id, face_id = filename.split("_")
                    values = [class_id, image_id, face_id]

                    all_values.append(", ".join(values) + "\n")
                    annotations.append(dict(zip(self.get_annotations_keys(), values)))

                open(annotation_path, "w").writelines(all_values)

        return annotations

    def _load_images(self) -> list:
        """
        Load the list of the images of the specified split

        :return: image paths
        """
        images = []
        filename = os.path.join(self.get_root_path(), f"{self._split}_list.txt")
        if os.path.exists(filename):
            images = [
                os.path.join(self.get_root_path(), *os.path.split(line.strip()))
                for line in open(filename).readlines()
            ]

        return images
