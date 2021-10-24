"""Collection of neural networks used in the module."""

import torch
from facenet_pytorch import InceptionResnetV1

from src.models.mobilenet import mobilenetv3


def mobilenet_v3(classify=False, mode="small", weights=None) -> torch.nn.Module():
    network = mobilenetv3(pretrained=False, classify=classify, n_class=8631, mode=mode)
    if weights:
        network.classify = True
        network.load_state_dict(torch.load(weights), strict=True)
        network.classify = classify
    return network


def teacher(classify=False) -> torch.nn.Module:
    return InceptionResnetV1(pretrained="vggface2", classify=classify)
