"""Collection of classes and functions for the VGGFace2 dataset"""

import os


class VGGFace2:
    """VGGFace2 main class"""

    ROOT_PATH = os.path.join("data", "raw", "vggface2")

    ANNOTATION_KEYS = ["class_id", "image_id", "face_id"]
    IS_AVAILABLE = os.path.exists(ROOT_PATH)

    def __init__(self, split: str):
        """
        Init function of the class

        :param split: split of the dataset, value must be either `train` or `test`
        """

        assert split in ["train", "test"]

        self._split = split
        self._path = os.path.join(self.ROOT_PATH, self._split)
        self._images = self._load_images()
        self._annotations = self._load_annotations()

    def _load_annotations(self) -> list:
        """
        Load the list of annotations of the defined split

        :return: image annotations
        """

        annotation_name = f"{self._split}_annotations.txt"
        annotation_path = os.path.join(self.ROOT_PATH, annotation_name)

        if os.path.exists(annotation_path):
            all_values = [line.strip() for line in open(annotation_path).readlines()]
            annotations = [
                dict(zip(self.ANNOTATION_KEYS, values.split(", ")))
                for values in all_values
            ]
        else:
            print(
                f"Generating (this is a one-time process) the file {annotation_name}..."
            )
            all_values = []
            annotations = []
            for image_path in self._images:
                class_id, filename = os.path.split(image_path)
                filename, _ = os.path.splitext(filename)
                image_id, face_id = filename.split("_")
                values = [class_id, image_id, face_id]

                all_values.append(", ".join(values) + "\n")
                annotations.append(dict(zip(self.ANNOTATION_KEYS, values)))

            open(annotation_path, "w").writelines(all_values)

        return annotations

    def _load_images(self) -> list:
        """
        Load the list of the images of the specified split

        :return: image paths
        """
        images = [
            os.path.join(self.ROOT_PATH, *os.path.split(line.strip()))
            for line in open(
                os.path.join(self.ROOT_PATH, f"{self._split}_list.txt")
            ).readlines()
        ]

        return images

    def get_image(self, index: int) -> str:
        """
        Get an image path given its index

        :param index: index of the image
        :return: path to the image
        :raises IndexError:
        """
        if index <= len(self._images) - 1:
            return self._images[index]

        raise IndexError(f"`{index} is not a valid index`")

    def get_image_annotations(self, index: int) -> dict:
        """
        Get the annotations of an image given its index

        :param index: index of the image
        :return: path to the image
        :raises IndexError:
        """
        if index <= len(self._annotations) - 1:
            return self._annotations[index]

        raise IndexError(f"`{index} is not a valid index`")
