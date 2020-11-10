# Face recognition

This repository contains the code and the results achieved during the time of my research project.

## Purpose

The purpose of the research is to develop a deep learning model able to recognize faces. The primary constraints of the
development are to build an architecture as small as possible that supports learning in production. Due to these
limits, in the repository are presented techniques for knowledge distillation and continual learning.

## Setup

### Installation

To initialize the environment, [conda](https://www.anaconda.com/products/individual#download-section) must be installed
on the system.

#### Clone this repository

```
git clone https://github.com/altndrr/face-recognition.git
```

#### Create the environment

With the usage of the `envinroment.yml` file, it is possible to create the environment.
```
conda env create -f environment.yml
```

### Download dataset

[VGGFace2](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) and [AgeDB](https://ibug.doc.ic.ac.uk/resources/agedb/) were
used to train and test the model. Both the datasets must be requested to the authors and cannot be redistributed.
Therefore, it is needed to retrieve them manually.

When both are available, their place should be in `data/raw/VGGFace2` and in `data/raw/AgeDB` respectively.

The structure of the `VGGFace2` dataset should be similar to the following.

```
├── VGGFace2
│   ├── test
│   │   ├── n000001
│   │   │   ├── 0001_01.jpg
│   │   │   ├── 0002_01.jpg
│   │   │   ├── ...
│   │   ├── n000009
│   │   │   ├── ...
│   │   ├── ...
│   ├── train
│   │   ├── n000002
│   │   │   ├── 0001_01.jpg
│   │   │   ├── 0002_01.jpg
│   │   │   ├── ...
│   │   ├── n000003
│   │   │   ├── ...
│   │   ├── ...
│   ├── test_list.txt
│   └── train_list.txt
```

While, the structure of the `AgeDB` dataset, should match the pattern below.
```
├── AgeDB
│   ├── 0_MariaCallas_35_f.jpg
│   ├── 1_MariaCallas_40_f.jpg
│   ├── ...
```

## Usage

The repository has two primary command-line interfaces, one for managing the project and the other for using it.

### Manage script

With the `manage.py` script, it is possible to manage the project.
```
python manage.py
```
Doing so with no option will display the `help` document.

Some examples of usage are:
1. `python manage.py code lint`: perform static analysis of the code.
2. `python manage.py code test`: run parallel tests with coverage.
3. `python manage.py env export`: generate a platform-independent `environment.yml` file;

### Source module

By typing the following line of code, it is possible to run the main module.
```
python -m src
```
As for the manage script, doing so with no option will display the `help` document.

## Structure

The repository is structured in five main folders:
1. `data`: comprises all of the data used, processed and outputted by the source code. 
2. `models`: includes all of the serialized models. 
3. `notebooks`: contains the notebooks for exploratory, communication and prototyping. 
4. `reports`: holds the documents with the results and the achievements.
5. `src`: carries all of the source code.

Five main modules compose the source code:
1. `data`: has all of the methods to transform and process the data.
2. `features`: comprehends all that is needed to manipulate the data and make them machine-consumable.
3. `models`: holds all of the code used to build and train models.
4. `tests`: includes the tests on the source code.
5. `visualization`: contains the methods used for image and video visualization.
