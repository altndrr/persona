"""Triplets command"""
import os
from glob import glob

from src.data import processed, raw
from src.features.generate import triplets
from src.utils import path
from src.utils.commands import Base


class Triplets(Base):
    """Triplets command class"""

    def run(self):
        if self.options["coverage"]:
            self.triplets_coverage()
        elif self.options["list"]:
            self.triplets_list()
        elif self.options["make"]:
            self.triplets_make()

    def triplets_coverage(self):
        """Evaluate the coverage of a triplet set"""
        print(f"Testing triplets coverage.")

        triplet_dataset = processed.TripletDataset(self.options["--set"])
        print(f"Test set composed of {len(triplet_dataset)} triplets.")

        total_classes = []
        total_paths = []
        for i in range(len(triplet_dataset)):
            paths, classes = triplet_dataset.get_triplet(i)
            total_classes.extend(classes)
            total_paths.extend(paths)
        total_classes = list(set(total_classes))
        total_paths = list(set(total_paths))
        print(
            f"Test dataset composed of {len(total_classes)} unique classes "
            f"and {len(total_paths)} unique images."
        )

        dataset_classes = None
        if triplet_dataset.get_name() == "agedb":
            original_dataset = raw.AgeDB()
            dataset_classes = 567
        elif triplet_dataset.get_name() == "lfw":
            original_dataset = raw.LFW()
            dataset_classes = 5479
        elif triplet_dataset.get_name() == "vggface2_test":
            original_dataset = raw.VGGFace2("test")
            dataset_classes = 500
        elif triplet_dataset.get_name() == "vggface2_train":
            original_dataset = raw.VGGFace2("train")
            dataset_classes = 8631
        dataset_images = len(original_dataset)
        print(
            f"Raw dataset composed of {dataset_classes} unique classes "
            f"and {dataset_images} unique images."
        )

        class_coverage = len(total_classes) / dataset_classes
        image_coverage = len(total_paths) / dataset_images
        print("Class coverage: %.3f." % class_coverage)
        print("Class coverage: %.3f." % image_coverage)

    def triplets_list(self):
        """List the generated triplets"""
        triplet_path = os.path.join(
            path.get_project_root(), "data", "processed", "triplets"
        )
        triplet_files = glob(os.path.join(triplet_path, "*.npy"))
        print_text = "\n - " + "\n - ".join(triplet_files)

        print(f"Triplets list: {print_text}")

    def triplets_make(self):
        """Make a new triplet file"""
        dataset = None
        if self.options["<dataset>"] == "agedb":
            dataset = raw.AgeDB()
        elif self.options["<dataset>"] == "lfw":
            dataset = raw.LFW()
        elif self.options["<dataset>"] == "vggface2" and self.options["--split"]:
            dataset = raw.VGGFace2(self.options["--split"])

        triplets(dataset, self.options["<num_triplets>"], self.options["--workers"])
