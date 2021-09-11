import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import torch
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
from random import shuffle
from tqdm import tqdm

from src.data.raw import AgeDB, CASIAWebFaces, LFW, VGGFace2
from src.features.transform import images_to_tensors
from src.utils import path

plt.style.use("ggplot")
plt.rcParams.update({"font.size": 14, "font.family": "Times New Roman"})


def agedb_age_scatter():
    """Draw the scatter plot of age of AgeDB."""
    dataset = AgeDB()
    ages = [int(ann["age"]) for _, ann in dataset]
    values = sorted(set(ages))
    counts = [ages.count(val) for val in values]

    fig = plt.figure(figsize=(10, 6), dpi=80)
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(values, counts)
    ax.set_xlim([0, 101])
    ax.set_ylim([0, 500])
    ax.set_ylabel("Number of pictures")
    ax.set_xlabel("Age")
    fig.tight_layout()


def dataset_picture_scatter():
    """Draw the scatter plot of pictures per identity for each primary dataset."""
    datasets = [AgeDB(), CASIAWebFaces(), LFW(), VGGFace2("train"), VGGFace2("test")]
    titles = ["AgeDB", "CASIA-WebFaces", "LFW", "VGGFace2 (train)", "VGGFace2 (test)"]

    fig = plt.figure(figsize=(10, 12), dpi=80)
    ax1 = fig.add_subplot(len(datasets), 1, 1)
    for i, dataset in enumerate(datasets):
        ax = fig.add_subplot(len(datasets), 1, i + 1, sharex=ax1, sharey=ax1)
        image_per_class = {}
        for _, ann in dataset:
            class_id = ann["class_id"]
            if class_id not in image_per_class:
                image_per_class[class_id] = 0
            image_per_class[class_id] += 1
        values = sorted(set(image_per_class.values()))
        counts = [list(image_per_class.values()).count(val) for val in values]

        ax.scatter(values, counts)
        ax.set_title(titles[i])
        ax.set_xlim([-10, 900])
        ax.set_ylim([-10, 600])
        ax.set_ylabel("Number of identities")

        if i + 1 == len(datasets):
            ax.set_xlabel("Number of pictures")

    fig.tight_layout()


def dataset_statistics():
    """Draw a bar graph of the statistics for each datasets."""
    datasets = [
        ["Youtube Faces", 0, 3425, 1595],
        ["WDRef", 99773, 0, 2995],
        ["VGGFace2", 3311286, 0, 9131],
        ["PubFig", 58797, 0, 200],
        ["LFW", 13233, 0, 5749],
        ["IJB-A", 5712, 2085, 500],
        ["CelebFaces+", 202599, 0, 10177],
        ["Casia-WebFaces", 494414, 0, 10575],
        ["AgeDB", 16516, 0, 568],
    ]

    df = pd.DataFrame(datasets, columns=["name", "images", "videos", "identities"])
    n_features = len(df.columns) - 1
    fig = plt.figure(figsize=(10, 12), dpi=80)
    for i, feature in enumerate(df.columns[1:]):
        ax = fig.add_subplot(n_features, 1, i + 1)
        ax.barh(df["name"], df[feature])
        ax.set_title(f"N. {feature} per dataset")
    fig.tight_layout()


def facenet_agedb_heatmap(n_classes=10):
    """Draw an heatmap of the performance of FaceNet on AgeDB."""
    BATCH_SIZE = 64
    dataset = AgeDB()

    classes = list(set([ann["class_id"] for _, ann in dataset]))
    shuffle(classes)
    classes = classes[:n_classes]

    images = []
    annotations = []
    for cls in classes:
        images.extend(dataset.get_images(class_id=cls))
        annotations.extend(dataset.get_annotations(class_id=cls))

    model = InceptionResnetV1(pretrained="vggface2").eval().to("cuda")
    dataloader = torch.utils.data.DataLoader(
        list(zip(images, annotations)),
        batch_size=BATCH_SIZE,
        num_workers=2,
        sampler=torch.utils.data.SequentialSampler(list(zip(images, annotations))),
    )

    embeddings = []
    class_ids = []
    for images, annotations in tqdm(dataloader):
        processed_images = []
        for i, image in enumerate(images):
            processed_image = path.change_data_category(image, "processed")
            if os.path.exists(processed_image):
                processed_images.append(processed_image)
                class_ids.append(annotations["class_id"][i])
        images = images_to_tensors(*processed_images)
        images = [fixed_image_standardization(image) for image in images]
        images = torch.stack(images).to("cuda")
        embeddings.extend(model(images).detach().to("cpu"))

    dists = []
    for e1 in tqdm(embeddings):
        curr_dists = [(e1 - e2).norm().item() for e2 in embeddings]
        dists.append(curr_dists)

    one_hot_labels = [classes.index(class_id) for class_id in class_ids]
    df = pd.DataFrame(dists, columns=one_hot_labels, index=one_hot_labels)
    average_df = (
        df.replace(0, None)
        .groupby(by=df.index)
        .mean()
        .groupby(by=df.columns, axis=1)
        .mean()
    )

    fig = plt.figure(figsize=(10, 8), dpi=80)
    ax = fig.add_subplot(1, 1, 1)
    sns.heatmap(average_df, ax=ax, cmap="magma")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

    return average_df
