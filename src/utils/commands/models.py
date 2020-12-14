"""Model command"""
import os
from glob import glob

import torch

from src.data import processed
from src.models import functions, nn
from src.utils import datasets as ds
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
        student = None
        if self.options["<model_name>"] == "mobilenet_v3_large":
            student = nn.mobilenet_v3(classify=True, mode="large")
        elif self.options["<model_name>"] == "mobilenet_v3_small":
            student = nn.mobilenet_v3(classify=True, mode="small")

        train_set, train_size = self._get_dataset(self.options["--train-set"])
        test_set, test_size = self._get_dataset(self.options["--test-set"])
        datasets = {"train": train_set, "test": test_set}

        print(f"Distilling model {type(student).__name__}.")
        print(f"Train set composed of {train_size} images.")
        print(f"Test set composed of {test_size} images.")
        print(f'Training for {self.options["--epochs"]} epochs.')
        print(f'Distillation temperature set to {self.options["--temperature"]}.')
        print(f'Training with {self.options["--lr"]} learning rate.')
        if not self.options["--no-lr-scheduler"]:
            print(f"Using MultiStep learning rate.")
        print(f'Using a batch size of {self.options["--batch"]}.')
        print(f'Using {self.options["--workers"]} workers.')

        student = functions.distill(
            student,
            datasets,
            temperature=self.options["--temperature"],
            batch_size=self.options["--batch"],
            epochs=self.options["--epochs"],
            lr=self.options["--lr"],
            num_workers=self.options["--workers"],
            no_lr_scheduler="--no-lr-scheduler" in self.options,
        )

        save_path = os.path.join(path.get_project_root(), "models")
        n_files = str(len(glob(os.path.join(save_path, "*.pth")))).zfill(2)
        epochs_to_string = str(self.options["--epochs"]).zfill(3)
        basename = f"{n_files}_{self.options['<model_name>']}_{epochs_to_string}.pth"
        filename = os.path.join(save_path, basename)
        print(f"Saved model to {filename}")
        student.eval().cpu()
        torch.save(student, filename)

        state_path = os.path.join(save_path, "dicts")
        os.makedirs(state_path, exist_ok=True)
        state_filename = os.path.join(state_path, basename)
        torch.save(student.state_dict(), state_filename)

    def models_list(self):
        """List the generated models"""
        models_path = os.path.join(path.get_project_root(), "models")
        triplet_files = glob(os.path.join(models_path, "*.pth"))
        print_text = "\n - " + "\n - ".join(triplet_files)

        print(f"Models list: {print_text}")

    def model_test(self):
        """Test a model's performance on a dataset"""
        model = None

        if self.options["--teacher"]:
            model = nn.teacher()
        elif self.options["--student"] is not None:
            model_path = os.path.join(path.get_project_root(), "models")
            model_files = glob(os.path.join(model_path, "*.pth"))

            model_file_id = str(self.options["--student"]).zfill(2)

            for file in model_files:
                if os.path.basename(file).startswith(model_file_id):
                    model = torch.load(file)
                    break

        if model is None:
            raise ValueError(f"{self.options['--student']} is an invalid file id")

        dataset, dataset_size = self._get_dataset(self.options["--set"])

        print(f"Testing model {type(model).__name__}.")
        print(f"Evaluating {self.options['--measure']} accuracy.")
        print(f"Test set composed of {dataset_size} images.")
        print(f'Using a batch size of {self.options["--batch"]}.')
        print(f'Using {self.options["--workers"]} workers.')

        functions.test(
            model,
            dataset,
            self.options["--measure"],
            batch_size=self.options["--batch"],
            num_workers=self.options["--workers"],
        )

    def _get_dataset(self, option):
        if option == "vggface2":
            dataset = ds.get_vggface2_dataset()
            dataset_size = len(dataset)
        elif option == "lfw":
            dataset = ds.get_lfw_dataset()
            dataset_size = len(dataset)
        else:
            dataset = processed.TripletDataset(option)
            dataset_size = len(dataset) * 3
        return dataset, dataset_size
