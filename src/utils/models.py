"""Collection of utility functions for the src.models module"""

from typing import Union

import numpy as np
import torch
from torch import optim
from torch.nn import functional

from src.utils import data


def get_distillation_lr_scheduler(
    epochs: int, optimizer: torch.optim
) -> Union[None, torch.optim.lr_scheduler.MultiStepLR]:
    """
    Get the learning rate scheduler for distillation.

    :param epochs: number of epochs of training
    :param optimizer: subject of scheduling
    :return:
    """
    scheduler_milestones = np.linspace(0, epochs, 5, dtype=np.int)[1:-1]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, scheduler_milestones)

    return scheduler


def get_distillation_temperatures(start_temperature: int, size: int, decay: str):
    """
    Generate an iterator containing the temperatures for each epoch.

    :param start_temperature: initial temperature
    :param size: number of temperature steps from the start to 1
    :param decay: modality of temperature decay, either constant or linear
    :return: iterator over temperature
    """
    assert decay in ["constant", "linear"]

    temperatures = None
    if decay == "constant":
        temperatures = (np.ones(size) * start_temperature).tolist()
    elif decay == "linear":
        temperatures = np.linspace(start_temperature, 1.0, size).tolist()

    return iter(temperatures)
