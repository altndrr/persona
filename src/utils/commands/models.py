"""Model command"""
import os
from glob import glob

import torch

from src.data import processed
from src.models import functions, nn
from src.utils import path
from src.utils.commands import Base


class Models(Base):
    """Model command class"""

    def run(self):
        if self.options["distill"]:
            self.model_distill()
        elif self.options["list"]:
            self.models_list()
        elif self.options["test"]:
            self.model_test()

    def model_distill(self):
        """Distill a model with the knowledge of a teacher"""
        try:
            self.options["<train_set_id>"] = int(self.options["<train_set_id>"])
            self.options["<test_set_id>"] = int(self.options["<test_set_id>"])
            self.options["<epochs>"] = int(self.options["<epochs>"])
            self.options["<temperature>"] = int(self.options["<temperature>"])
        except ValueError as value_error:
            raise value_error

        decay = "constant"
        if self.options["--decay"]:
            if self.options["<decay_type>"] not in ["constant", "linear"]:
                raise ValueError(
                    f'{self.options["<decay_type>"]} is an invalid decay type.'
                    f"Select between `constant` or `linear`"
                )
            decay = self.options["<decay_type>"]

        if self.options["<model_name>"] == "mobilenet_v2":
            student = nn.mobilenet_v2()
        else:
            raise ValueError(f'{self.options["<model_name>"]} is an invalid student')

        datasets = {
            "train": processed.TripletDataset(
                self.options["<train_set_id>"], transforms=True
            ),
            "test": processed.TripletDataset(self.options["<test_set_id>"]),
        }

        print(f"Distilling model {type(student).__name__}.")
        print(f'Train set composed of {len(datasets["train"])} triplets.')
        print(f'Test set composed of {len(datasets["test"])} triplets.')
        print(f'Training for {self.options["<epochs>"]} epochs.')
        print(f'Start temperature {self.options["<temperature>"]}, decay {decay}.')
        if "--no-lr-scheduler" in self.options:
            print(f"Using MultiStep learning rate")

        student = functions.distill(
            student,
            datasets,
            initial_temperature=self.options["<temperature>"],
            temperature_decay=decay,
            batch_size=16,
            epochs=self.options["<epochs>"],
            num_workers=8,
            no_lr_scheduler="--no-lr-scheduler" in self.options,
        )

        save_path = os.path.join(path.get_project_root(), "models")
        n_files = str(len(glob(os.path.join(save_path, "*.pth")))).zfill(2)
        epochs_to_string = str(self.options["<epochs>"]).zfill(3)
        basename = f"{n_files}_{self.options['<model_name>']}_{epochs_to_string}.pth"
        filename = os.path.join(save_path, basename)
        print(f"Saved model to {filename}")
        torch.save(student, filename)

    def models_list(self):
        """List the generated models"""
        models_path = os.path.join(path.get_project_root(), "models")
        triplet_files = glob(os.path.join(models_path, "*.pth"))
        print_text = "\n - " + "\n - ".join(triplet_files)

        print(f"Triplets list: {print_text}")

    def model_test(self):
        """Test a model's performance on a dataset"""

        model = None

        if self.options["teacher"]:
            model = nn.teacher()
        elif self.options["<model_id>"]:
            model_path = os.path.join(path.get_project_root(), "models")
            model_files = glob(os.path.join(model_path, "*.pth"))

            model_file_id = str(self.options["<model_id>"]).zfill(2)

            for file in model_files:
                if os.path.basename(file).startswith(model_file_id):
                    model = torch.load(file)
                    break

            if model is None:
                raise ValueError(f"{self.options['<model_id >']} is an invalid file id")
        else:
            raise ValueError(f'{self.options["<model_name>"]} is an invalid student')

        dataset = processed.TripletDataset(self.options["<test_set_id>"])

        print(f"Testing model {type(model).__name__}.")
        print(f"Test set composed of {len(dataset)} triplets.")

        functions.test(model, dataset)
