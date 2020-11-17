"""Triplets command"""
import os
from glob import glob

from src.data import raw
from src.features.generate import triplets
from src.utils import path
from src.utils.commands import Base


class Triplets(Base):
    """Triplets command class"""

    def run(self):
        if self.options["list"]:
            self.triplets_list()
        elif self.options["make"]:
            self.triplets_make()

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
        try:
            self.options["<number>"] = int(self.options["<number>"])
            self.options["<processes>"] = int(self.options["<processes>"])
        except ValueError as value_error:
            raise value_error

        if self.options["<processes>"] == 0:
            self.options["<processes>"] = os.cpu_count()

        if self.options["<dataset>"] == "agedb":
            dataset = raw.AgeDB()
        elif self.options["<dataset>"] == "vggface2" and self.options["<split>"]:
            dataset = raw.VGGFace2(self.options["<split>"])
        else:
            raise ValueError()

        triplets(dataset, self.options["<number>"], self.options["<processes>"])
