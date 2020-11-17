"""Model command"""
import os
from glob import glob

import torch
from facenet_pytorch import InceptionResnetV1

from src.data import processed
from src.models import functions, nn
from src.utils import path
from src.utils.commands import Base


class Models(Base):
    """Model command class"""

    def run(self):
        if self.options["distill"]:
            self.model_distill()
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

        print(f'Distilling model {self.options["<model_name>"]}.')
        print(
            f'Training on {len(datasets["train"])} triplets, testing on {len(datasets["test"])}.'
        )
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
        basename = f"{n_files}_{self.options['<model_name>']}_{str(self.options['<epochs>']).zfill(3)}.pth"
        filename = os.path.join(save_path, basename)
        print(f"Saving model to {filename}...")
        torch.save(student.state_dict(), filename)

    def model_test(self):
        """Test a model's perfomance on a dataset"""
        if self.options["<model_name>"] == "teacher":
            model = InceptionResnetV1(pretrained="vggface2", classify=True)
        else:
            raise ValueError(f'{self.options["<model_name>"]} is an invalid student')

        dataset = processed.TripletDataset(self.options["<test_set_id>"])

        print(
            f'Testing model {self.options["<model_name>"]} on {len(dataset)} triplets.'
        )

        functions.test(model, dataset)
