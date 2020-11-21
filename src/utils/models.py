"""Collection of utility functions for the src.models module"""

from typing import Union

import numpy as np
import torch
from torch import optim


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
