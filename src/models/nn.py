"""Collection of mobilenet implementations"""

import torch
from torchvision.models import mobilenet_v2


def mobilenet_v2():
    student = mobilenet_v2(pretrained=False)

    student.classifier = torch.nn.Sequential(
        # Keep the same dropout as of the base mobilenet.
        torch.nn.Dropout(p=0.2, inplace=False),
        # Add a linear that matches the out_features of the teacher.
        torch.nn.Linear(in_features=1280, out_features=512, bias=False),
        # Add a batch norm as in the teacher.
        torch.nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True),
        # Add a last linear that matches the number of classes in VGGFace2.
        torch.nn.Linear(512, 8631),
    )

    return student