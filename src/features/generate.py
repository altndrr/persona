"""Collection of functions to generate data"""

import multiprocessing
import os
import random
from glob import glob

import numpy as np
from torchvision.utils import save_image
from tqdm import tqdm

from src.data.raw import Raw
from src.features import transform
from src.utils import path


def _triplets(dataset: Raw, n_triplets: int, process_id: int):
    classes = list(set([annotation["class_id"] for image, annotation in dataset]))
    triplet_list = []

    progress_bar = tqdm(range(int(n_triplets)))

    for _ in progress_bar:
        pos_class = random.choice(classes)
        pos_images = dataset.get_images(class_id=pos_class)
        while len(pos_images) < 2:
            pos_class = random.choice(classes)
            pos_images = dataset.get_images(class_id=pos_class)

        neg_class = random.choice(classes)
        while neg_class == pos_class:
            neg_class = random.choice(classes)

        anchor = random.choice(pos_images)
        positive = random.choice(pos_images)
        while anchor == positive:
            positive = random.choice(pos_images)

        negative = dataset.get_images(n_random=1)
        while negative in pos_images:
            negative = dataset.get_images(n_random=1)

        aligned_images = transform.align_images(anchor, positive, negative)

        if len(aligned_images) == 3:
            triplet_list.append([anchor, positive, negative, pos_class, neg_class])

            for aligned_image, original_path in zip(
                aligned_images, [anchor, positive, negative]
            ):
                abs_original_path = os.path.join(path.get_project_root(), original_path)
                save_path, save_name = os.path.split(abs_original_path)
                os.makedirs(save_path, exist_ok=True)
                save_image(
                    aligned_image, path.change_data_category(original_path, "processed")
                )

    # Update the total value of triplets, since some could have been discarded (i.e. not able to
    # align, or other reason)
    n_triplets = len(triplet_list)

    temp_path = path.change_data_category(dataset.get_path(), "interim")
    os.makedirs(temp_path, exist_ok=True)
    np.save(
        os.path.join(temp_path, f"triplets_{n_triplets}_{process_id}.npy"), triplet_list
    )

    return triplet_list


def triplets(dataset: Raw, n_triplets: int, n_processes: int):
    """Generate a set of triplets. It saves an aligned copy of each image in a triplet in
    `data/processed` and a file `.npy` containing the list of triplets.

    :param dataset: dataset to use to generate triplets
    :param n_triplets: number o triplets to generate
    :param n_processes: number of processes to use
    """
    # NOTE: code is inspired by https://github.com/tamerthamoqa/facenet-pytorch-vggface2
    triplet_list = []

    print(f"Generating {n_triplets} triplets using {n_processes} processes...")

    triplet_residual = n_triplets % n_processes

    if triplet_residual == 0:
        n_triplets_per_process = n_triplets / n_processes
    else:
        n_triplets_per_process = (n_triplets - triplet_residual) / n_processes

    processes = []
    for i in range(n_processes):
        processes.append(
            multiprocessing.Process(
                target=_triplets, args=(dataset, n_triplets_per_process, i)
            )
        )

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    if triplet_residual != 0:
        _triplets(dataset, triplet_residual, n_processes + 1)

    temp_path = path.change_data_category(dataset.get_path(), "interim")
    numpy_files = glob(os.path.join(temp_path, "*.npy"))

    for numpy_file in numpy_files:
        triplet_list.extend(np.load(numpy_file))
        os.remove(numpy_file)

    # Update the total value of triplets, since some could have been discarded (i.e. not able to
    # align, or other reason)
    n_triplets = len(triplet_list)

    file_path = os.path.dirname(__file__)
    save_path = os.path.join(path.get_project_root(), "data", "processed", "triplets")
    os.makedirs(save_path, exist_ok=True)

    n_files = str(len(glob(os.path.join(save_path, "*.*")))).zfill(2)
    basename = f"{n_files}_{dataset.get_name()}_{n_triplets}.npy"
    filename = os.path.join(save_path, basename)

    print(f"Saving triplets to {filename}...")
    np.save(filename, triplet_list)
