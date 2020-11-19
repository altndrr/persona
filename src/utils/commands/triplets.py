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
        print(f"Test set composed of {len(total_classes)} unique classes.")
        print(f"Test set composed of {len(total_paths)} unique images.")

        original_dataset = None
        if triplet_dataset.get_name() == "agedb":
            original_dataset = raw.AgeDB()
        elif triplet_dataset.get_name() == "lfw":
            original_dataset = raw.LFW()
        elif triplet_dataset.get_name() == "vggface2_test":
            original_dataset = raw.VGGFace2("test")
        elif triplet_dataset.get_name() == "vggface2_train":
            original_dataset = raw.VGGFace2("train")
        print(f"Raw dataset composed of {len(original_dataset)} unique images.")

        image_coverage = len(total_paths) / len(original_dataset)
        print(f"Image coverage of {image_coverage}.")

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
