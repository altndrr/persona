"""Collection of functions to apply on neural networks"""
import os
from glob import glob
from typing import Dict, Union

import numpy as np
import torch
import torch.utils.data
import torchvision

from src.data import processed
from src.models import nn
from src.utils import data, lfw, models, path


def distill(
    student: torch.nn.Module,
    datasets: Dict[str, processed.TripletDataset],
    temperature: int = 10,
    batch_size: int = 16,
    epochs: int = 10,
    lr: float = 0.001,
    num_workers: int = 8,
    no_lr_scheduler: bool = False,
) -> torch.nn.Module:
    """
    Distill a student network with the knowledge of an InceptionResnetV1 teacher.

    :param student: student network
    :param datasets: dictionary containing the train and test datasets
    :param temperature: temperature of distillation
    :param batch_size: size of the batch for the train and test data loaders
    :param epochs: number of epochs to train
    :param lr: learning rate of the network
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

    student = student.to(device)
    teacher = nn.teacher(classify=True).eval().to(device)

    train_loader = data.get_triplet_dataloader(
        datasets["train"], batch_size, num_workers
    )
    test_loader = data.get_triplet_dataloader(datasets["test"], batch_size, num_workers)

    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    scheduler = (
        None
        if no_lr_scheduler
        else models.get_distillation_lr_scheduler(epochs, optimizer)
    )

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}")

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
        print("Match accuracy: %.3f" % accuracy)

        if scheduler:
            scheduler.step()

    return student


def test(
    network: torch.nn.Module,
    dataset: Union[processed.TripletDataset, torchvision.datasets.ImageFolder],
    measure: str,
    batch_size: int = 16,
    num_workers: int = 8,
):
    """
    Test a network on a specific dataset.

    :param network: neural network to test
    :param dataset: collection of data to test
    :param batch_size: size of the batch for the train and test data loaders
    :param measure: measure to test, either class or match accuracy
    :param num_workers: number of workers to use for each data loader
    :return:
    """
    if measure not in ["class", "match"]:
        raise ValueError(f"{measure} is an invalid test measure")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    network = network.to(device)

    if isinstance(dataset, processed.TripletDataset):
        loader = data.get_triplet_dataloader(dataset, batch_size, num_workers)

        if measure == "class" and loader.dataset.get_name() != "vggface2_train":
            raise ValueError(
                f"class accuracy can only be measured on a vggface2 train dataset"
            )

        accuracy = 0
        if measure == "class":
            accuracy = test_class_accuracy(network, loader, device)
        elif measure == "match":
            accuracy = test_match_accuracy(network, loader, device)

        print("%s accuracy: %.3f" % (measure.title(), accuracy))
    elif isinstance(dataset, torchvision.datasets.ImageFolder):
        if measure == "class":
            raise ValueError(f"class accuracy cannot be measured on lfw dataset")
        loader = data.get_image_dataloader(dataset, batch_size, num_workers)
        accuracy = test_lfw(network, loader, device)
        print("LFW mean accuracy: %.3f" % accuracy)


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
    network.classify = True

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
    network.classify = False

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
    student.classify = True

    running_loss = 0.0
    running_soft_loss = 0.0
    running_hard_loss = 0.0

    count = 0
    for _, (inputs, labels) in enumerate(loader):
        # Convert the class names with the class id and transpose the tensor.
        labels = [classes.index(label) for label in labels]
        labels = torch.LongTensor(labels).T

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        student_outputs = student(inputs)
        teacher_outputs = teacher(inputs)

        # Evaluate the soft loss.
        soft_outputs = torch.nn.functional.log_softmax(
            student_outputs / temperature, dim=1
        )
        soft_targets = torch.nn.functional.softmax(teacher_outputs / temperature, dim=1)
        soft_loss = torch.nn.functional.kl_div(
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
        soft_weight, hard_weight = 0.9, 0.1
        loss = soft_loss * soft_weight + hard_loss * hard_weight

        loss.backward()
        optimizer.step()

        # Print statistics.
        running_loss += loss.item()
        running_soft_loss += soft_loss.item() * soft_weight
        running_hard_loss += hard_loss.item() * hard_weight
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


def test_lfw(network, loader, device):
    """
    Test the class accuracy of a network on a dataset.

    :param network: network to test
    :param loader: loader to test
    :param device: device to use
    :return: result accuracy
    """
    network.eval()
    network.classify = False

    data_dir = os.path.join(path.get_project_root(), "data", "processed", "lfw")
    pairs_path = os.path.join(
        path.get_project_root(), "data", "raw", "lfw", "pairs.txt"
    )

    paths = sorted(glob(os.path.join(data_dir, "*", "*.jpg")))

    embeddings = []
    for xb, yb in loader:
        xb = xb.to(device)
        b_embeddings = network(xb)
        b_embeddings = b_embeddings.detach().to("cpu").numpy()
        embeddings.extend(b_embeddings)
    embeddings_dict = dict(zip(paths, embeddings))

    pairs = lfw.read_pairs(pairs_path)
    path_list, issame_list = lfw.get_paths(data_dir, pairs)
    embeddings = np.array([embeddings_dict[path] for path in path_list])

    tpr, fpr, accuracy, val, val_std, far, fp, fn = lfw.evaluate(
        embeddings, issame_list
    )

    return np.mean(accuracy)
