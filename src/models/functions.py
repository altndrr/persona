"""Collection of functions to apply on neural networks"""

import os
from glob import glob
from typing import Dict

import numpy as np
import torch
import torch.utils.data
from torch import optim
from torch.nn import functional

from src.data.processed import TripletDataset
from src.models import nn
from src.utils import path


def distill(
    student: torch.nn.Module,
    datasets: Dict[str, TripletDataset],
    initial_temperature: int = 10,
    temperature_decay: str = "linear",
    batch_size: int = 16,
    epochs: int = 10,
    num_workers: int = 8,
    no_lr_scheduler: bool = False,
) -> torch.nn.Module:
    """
    Distill a student network with the knowledge of an InceptionResnetV1 teacher.

    :param student: student network
    :param datasets: dictionary containing the train and test datasets
    :param initial_temperature: initial temperature for distillation
    :param temperature_decay: either constant or linear, defines how temperature changes over time
    :param batch_size: size of the batch for the train and test data loaders
    :param epochs: number of epochs to train
    :param num_workers: number of workers to use for each data loader
    :param no_lr_scheduler: don't reduce the learning rate at 1/3 and 2/3 of the training
    :return: distilled student
    """

    if datasets["train"].get_name() != "vggface2_train":
        raise ValueError(
            "the train dataset must be based on the train partition of the vggface2 raw dataset"
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    student.to(device)
    student.train()

    teacher = nn.teacher()
    teacher.to(device)
    teacher.eval()

    loaders = {
        "train": torch.utils.data.DataLoader(
            datasets["train"],
            batch_size=batch_size,
            collate_fn=TripletDataset.collate_fn,
            num_workers=num_workers,
        ),
        "test": torch.utils.data.DataLoader(
            datasets["test"],
            batch_size=batch_size,
            collate_fn=TripletDataset.collate_fn,
            num_workers=num_workers,
        ),
    }
    optimizer = optim.Adam(student.parameters(), lr=0.001)

    if no_lr_scheduler:
        scheduler = None
    else:
        scheduler_milestones = np.linspace(0, epochs, 5, dtype=np.int)[1:-1]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, scheduler_milestones)

    classes = [
        os.path.basename(folder)
        for folder in glob(
            os.path.join(
                path.get_project_root(), "data", "raw", "vggface2", "train", "*"
            )
        )
    ]

    def get_temperatures(start_temperature, size, decay):
        assert decay in ["constant", "linear"]

        if decay == "constant":
            return (np.ones(size) * start_temperature).tolist()
        elif decay == "linear":
            return np.linspace(start_temperature, 1.0, size).tolist()

    temperatures = get_temperatures(initial_temperature, epochs, temperature_decay)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}, temperature = {temperatures[epoch]}")

        print("Training:")
        _train_step(
            student,
            teacher,
            classes,
            loaders["train"],
            temperatures[epoch],
            optimizer,
            epoch,
            device,
        )

        if scheduler:
            scheduler.step()

        print("Testing:")
        _test_step(student, loaders["test"], device)

    return student


def test(
    network: torch.nn.Module,
    dataset: TripletDataset,
    batch_size: int = 16,
    num_workers: int = 8,
):
    """
    Test a network on a specific dataset.

    :param network: neural network to test
    :param dataset: dictionary containing the train and test datasets
    :param batch_size: size of the batch for the train and test data loaders
    :param num_workers: number of workers to use for each data loader
    :return:
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    network.to(device)
    network.eval()

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=TripletDataset.collate_fn,
        num_workers=num_workers,
    )

    _test_step(network, loader, device)


def _test_step(model, loader, device):
    model.eval()

    match_accuracy = 0

    for _, sample in enumerate(loader):
        inputs, _ = sample
        inputs = inputs.to(device)

        outputs = model(inputs)

        # Evaluate match accuracy.
        for i in range(0, len(inputs), 3):
            triplet = outputs[i : i + 3]
            distance = [
                (triplet[0] - triplet[1]).norm().item(),
                (triplet[0] - triplet[2]).norm().item(),
            ]
            if distance[0] < distance[1]:
                match_accuracy += 1

    print(f"Match accuracy: {match_accuracy / len(loader.dataset)}")


def _train_step(
    student,
    teacher,
    classes,
    loader,
    temperature,
    optimizer,
    epoch,
    device,
    print_every=100,
):
    student.train()

    running_loss = 0.0
    running_soft_loss = 0.0
    running_hard_loss = 0.0

    count = 0
    for _, sample in enumerate(loader):
        soft_loss = 0
        hard_loss = 0

        inputs, labels = sample

        n_triplets = len(inputs) / 3

        # Convert the class names with the class id and transpose the tensor.
        labels = [classes.index(label) for label in labels]
        labels = torch.LongTensor(labels).T

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        student_outputs = student(inputs)
        teacher_outputs = teacher(inputs)

        # Evaluate the soft loss.
        for i in range(0, len(inputs), 3):
            soft_outputs = functional.log_softmax(
                student_outputs[i : i + 3] / temperature, dim=1
            )
            soft_targets = functional.softmax(
                teacher_outputs[i : i + 3] / temperature, dim=1
            )
            soft_loss += functional.kl_div(
                soft_outputs, soft_targets.detach(), reduction="batchmean"
            )

        # It is important to multiply the soft loss by T^2 when using both hard and soft
        # targets. This ensures that the relative contributions of the hard and soft
        # targets remain roughly unchanged if the temperature used for distillation is
        # changed while experimenting with meta-parameters.
        soft_loss *= temperature ** 2

        # Evaluate the hard loss.
        for i in range(0, len(inputs), 3):
            hard_loss += functional.cross_entropy(
                student_outputs[i : i + 3], labels[i : i + 3]
            )

        # Evaluate the weighted average loss.
        loss = soft_loss + hard_loss

        loss.backward()
        optimizer.step()

        # Print statistics.
        running_loss += loss.item() / n_triplets
        running_soft_loss += soft_loss.item() / n_triplets
        running_hard_loss += hard_loss.item() / n_triplets
        if count % print_every == print_every - 1:
            print(
                "[%d, %5d] loss: %.3f (soft: %.3f, hard: %.3f)"
                % (
                    epoch + 1,
                    count + 1,
                    running_loss / print_every,
                    running_soft_loss / print_every,
                    running_hard_loss / print_every,
                )
            )
            running_loss = 0.0
            running_soft_loss = 0.0
            running_hard_loss = 0.0

        count += 1
