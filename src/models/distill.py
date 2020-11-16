"""Collection of functions to distill knowledge from facenet to a student"""

import torch
import torch.utils.data
from torch.nn import functional


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
