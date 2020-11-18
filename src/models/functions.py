"""Collection of functions to apply on neural networks"""

from typing import Dict

import torch
import torch.utils.data

from src.data import processed
from src.models import nn
from src.utils import data, models


def distill(
    student: torch.nn.Module,
    datasets: Dict[str, processed.TripletDataset],
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
    classes = data.get_vggface2_classes("train")
    temperatures = models.get_distillation_temperatures(
        initial_temperature, epochs, temperature_decay
    )

    student = student.train().to(device)
    teacher = nn.teacher().eval().to(device)

    train_loader = data.get_triplet_dataloader(
        datasets["train"], batch_size, num_workers
    )
    test_loader = data.get_triplet_dataloader(datasets["test"], batch_size, num_workers)

    optimizer = torch.optim.Adam(student.parameters(), lr=0.001)
    scheduler = (
        None
        if no_lr_scheduler
        else models.get_distillation_lr_scheduler(epochs, optimizer)
    )

    for epoch in range(epochs):
        temperature = next(temperatures)
        print(f"\nEpoch {epoch + 1}, temperature = {temperature}")

        print("Training:")
        distill_step(
            student,
            teacher,
            classes,
            train_loader,
            temperature,
            optimizer,
            epoch,
            device,
        )

        print("Testing:")
        accuracy = test_match_accuracy(student, test_loader, device)
        print(f"Match accuracy: {accuracy}")

        if scheduler:
            scheduler.step()

    return student


def test(
    network: torch.nn.Module,
    dataset: processed.TripletDataset,
    measure: str,
    batch_size: int = 16,
    num_workers: int = 8,
):
    """
    Test a network on a specific dataset.

    :param network: neural network to test
    :param dataset: dictionary containing the train and test datasets
    :param batch_size: size of the batch for the train and test data loaders
    :param measure: measure to test, either class or match accuracy
    :param num_workers: number of workers to use for each data loader
    :return:
    """
    if measure not in ["class", "match"]:
        raise ValueError(f"{measure} is an invalid test measure")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    network = network.eval().to(device)

    loader = data.get_triplet_dataloader(dataset, batch_size, num_workers)

    if measure == "class" and loader.dataset.get_name() != "vggface2_train":
        raise ValueError(
            f"class accuracy can only me measured on a vggface2 train dataset"
        )

    accuracy = 0
    if measure == "class":
        accuracy = test_class_accuracy(network, loader, device)
    elif measure == "match":
        accuracy = test_match_accuracy(network, loader, device)

    print(f"{measure.title()} accuracy: {accuracy}")


def test_class_accuracy(
    network: torch.nn, loader: torch.utils.data.DataLoader, device: torch.device
) -> float:
    """
    Test the class accuracy of a network on a dataset.

    :param network: network to test
    :param loader: loader to test
    :param device: device to use
    :return: result accuracy
    """
    network.eval()

    accuracy = 0
    classes = data.get_vggface2_classes("train")

    for _, sample in enumerate(loader):
        inputs, labels = sample
        inputs = inputs.to(device)

        outputs = network(inputs)

        # Convert the class names with the class id and transpose the tensor.
        labels = [classes.index(label) for label in labels]
        labels = torch.LongTensor(labels).T
        labels = labels.to(device)

        output_labels = torch.topk(outputs, 1).indices.view(-1)
        accuracy += torch.count_nonzero(output_labels == labels)

    accuracy = accuracy / (len(loader.dataset) * 3)
    return accuracy


def test_match_accuracy(network, loader, device) -> float:
    """
    Test the match accuracy of a network on a dataset.

    :param network: network to test
    :param loader: loader to test
    :param device: device to use
    :return: result accuracy
    """
    network.eval()

    accuracy = 0

    for _, (inputs, labels) in enumerate(loader):
        inputs = inputs.to(device)

        outputs = network(inputs)

        for i in range(0, len(inputs), 3):
            triplet = outputs[i : i + 3]
            distance = [
                (triplet[0] - triplet[1]).norm().item(),
                (triplet[0] - triplet[2]).norm().item(),
            ]
            if distance[0] < distance[1]:
                accuracy += 1

    accuracy /= len(loader.dataset)
    return accuracy


def distill_step(
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
    """
    Perform a single step of distillation.

    :param student: student network
    :param teacher: teacher network
    :param classes: classes of the dataset
    :param loader: loader used for training
    :param temperature: current temperature of distillation
    :param optimizer: optimizer to use
    :param epoch: current epoch
    :param device: device to use
    :param print_every: print a progress statement every defined number of batches
    :return:
    """
    student.train()

    running_loss = 0.0
    running_soft_loss = 0.0
    running_hard_loss = 0.0

    count = 0
    for _, (inputs, labels) in enumerate(loader):
        n_triplets = len(inputs) / 3

        # Convert the class names with the class id and transpose the tensor.
        labels = [classes.index(label) for label in labels]
        labels = torch.LongTensor(labels).T

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        student_outputs = student(inputs)
        teacher_outputs = teacher(inputs)

        # Evaluate the soft loss.
        soft_loss = 0
        for i in range(0, len(inputs), 3):
            soft_outputs = torch.nn.functional.log_softmax(
                student_outputs[i : i + 3] / temperature, dim=1
            )
            soft_targets = torch.nn.functional.softmax(
                teacher_outputs[i : i + 3] / temperature, dim=1
            )
            soft_loss += torch.nn.functional.kl_div(
                soft_outputs, soft_targets.detach(), reduction="batchmean"
            )

        # It is important to multiply the soft loss by T^2 when using both hard and soft
        # targets. This ensures that the relative contributions of the hard and soft
        # targets remain roughly unchanged if the temperature used for distillation is
        # changed while experimenting with meta-parameters.
        soft_loss *= temperature ** 2

        # Evaluate the hard loss.
        hard_loss = torch.nn.functional.cross_entropy(student_outputs, labels)

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
