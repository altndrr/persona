# Persona

This repository contains the code and the results of my research project in deep learning.

## Purpose

The purpose of the research is to develop a deep learning model able to recognize faces. The primary constraints of the
development are to build an architecture as small as possible that supports learning in production. Due to these
limits, in the repository are presented techniques for knowledge distillation and continual learning.

## Setup

To initialize the environment, [conda](https://www.anaconda.com/products/individual#download-section) must be installed
on the system.

#### Clone the repository

```
git clone https://github.com/altndrr/persona.git
```

#### Create the environment

With the usage of the `environment.yml` file, it is possible to create the environment.
```
conda env create -f environment.yml
```

### Download the datasets

[AgeDB](https://ibug.doc.ic.ac.uk/resources/agedb/), [LFW](http://vis-www.cs.umass.edu/lfw/) and
[VGGFace2](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) were used to train and test the
models. AgeDB and VGGFace2 must be requested to the authors and cannot be redistributed. On the other
hand, LFW is available for free. All three of them are to be retrieved manually.

When all of them are available, their place should be in the `data/raw/` folder, under their
respective names.

AgeDB has all of the images in the same folder, while LFW and VGGFace2 have them divided in subfolder
representing their identities. Moreover, VGGFace2 is split into test and train. Therefore, the
`data/raw` folder should be similar to the following.
```
├── AgeDB
│   ├── 0_MariaCallas_35_f.jpg
│   ├── 1_MariaCallas_40_f.jpg
│   └── ...
├── LFW
│   ├── Aaron_Eckhart
│   │   │   └── Aaron_Eckhart_0001.jpg
│   ├── Aaron_Guiel
│   ├── ...
│   └── pairs.txt
├── VGGFace2
│   ├── test
│   │   ├── n000001
│   │   │   ├── 0001_01.jpg
│   │   │   ├── 0002_01.jpg
│   │   │   └── ...
│   │   ├── n000009
│   │   └── ...
│   ├── train
│   │   ├── n000002
│   │   ├── n000003
│   │   └── ...
│   ├── test_list.txt
│   └── train_list.txt
```

Note that in the VGGFace2 folder there are also a `test_list.txt` and a `train_list.txt` files that
are required. The same reasoning goes for the `pairs.txt` file in the LFW folder. All of the files
can be retrieved on the datasets' website.

## Usage

The repository has two primary command-line interfaces, one for managing the project and the other
for using it.

### Main module

By typing the following line of code, it is possible to run the main module.
```
python -m src
```
Doing so with no options will display the `help` document.
```
Usage:
    main models distill <model_name> --train-set=[<ID> | vggface2] --test-set=[<ID> | lfw] -e <NUM> -t <VAL> [--lr=<VAL>] [--no-lr-scheduler] [-b SIZE] [-w NUM]
    main models list
    main models test (--student=<ID> | --teacher) --set=<ID> [--measure=<VAL>] [-b SIZE] [-w NUM]
    main triplets coverage --set=<ID>
    main triplets list
    main triplets make <num_triplets> <dataset> [--split=<VAL>] [-w <NUM>]
    main -h | --help
    main --version

Options:
    -b SIZE --batch=<SIZE>          Batch size [default: 16].
    -e NUM --epochs=<NUM>           Number of epochs to train.
    -t NUM --temperature=<NUM>      Temperature for distillation.
    -w NUM --workers=<NUM>          Number of workers [default: 8].
    --lr=<VAL>                      Learning rate for training [default: 0.001].
    --measure=<VAL>                 Test measure to use, either class or match [default: match].
    --no-lr-scheduler               Don't use a learning rate scheduler.
    --set=[<ID> | lfw | vggface2]   ID of a triplet (either for training or for testing) or dataset name.
    --split=<VAL>                   Split of the dataset, either train or test.
    --student=<ID>                  ID of the student network.
    --train-set=[<ID> | vggface2]   ID of the training set or dataset name.
    --test-set=[<ID> | lfw]         ID of the testing set or dataset name.
    -h --help                       Show this screen.
    --version                       Show version.
```

### Manage script

With the `manage.py` script, it is possible to manage the project.
The primary commands are below.
1. `python manage.py code --format`: format code according to `.pylintrc`.
2. `python manage.py code --lint`: perform static analysis of the code.
3. `python manage.py code --test`: run parallel tests with coverage.

## Structure

The repository is structured in five main folders:
1. `data`: comprises all of the data used, processed and outputted by the source code. 
2. `models`: includes all of the serialized models. 
3. `notebooks`: contains the notebooks for exploratory, communication and prototyping. 
4. `reports`: holds the documents with the results and the achievements.
5. `src`: carries all of the source code.

Five main modules compose the source code:
1. `data`: has all the classes and the functions needed to work with data and datasets.
2. `features`: comprehends the methods to transform and process the data.
3. `models`: holds all of the code used to build and train models.
4. `tests`: includes the tests on the source code.
5. `visualization`: contains the methods used for image and video visualization.

## Workflow
The main operations that can be performed are described below. They are presented in a logical order
to guide the reader through a complete workflow.

### Add a new raw dataset
Given a dataset in its raw format, by simply adding a subclass of the `data.raw.Raw` class, it 
is possible to add full support to it in the module. By default, the module support AgeDB, LFW and
VGGFace2.

### Process a triplet dataset
Models aren't supposed to work directly on raw datasets: it is needed to process them. Either for
testing the performance or for distillation, neural networks are supposed to work on triplet
datasets. An item of a triplet dataset is made of three images (i.e. an anchor, a positive and a
negative), and their respective classes. Anchor and positive share the same class, while negative
do not.

The choice to use triplet datasets as a base structure was born by the will to cover most of the
existing training processes for face recognition. Indeed, if triplet loss is used for training, the
support is implicit in the dataset structure. Moreover, by simply "flattening" a triplet, it is
still possible to e.g. train a network as a classifier to solve face recognition. Lastly, the
triplet format also guarantees a fast way to test match accuracy.

To process a raw dataset into a triplet dataset the following command is available.
```
python -m src triplets make <num_triplets> <dataset> [--split=<VAL>] [-w <NUM>]
```

Moreover, it is possible to check the generated set coverage with the following line.
```
python -m src triplets coverage --set=<ID>
```

### Distill
Knowledge distillation can be performed to pass knowledge from a teacher model to a student. Usually,
it's performed on a student of smaller dimension. Here, the students are different implementations of
MobileNet. Moreover, the teacher model is fixed and is the FaceNet implementation of timesler
(see References). This implementation support two variants, one trained on CASIA-Webface, the other
on VGGFace2. For the time being, this module only supports distillation from the VGGFace2 version.

To distill knowledge, both a train and a test triplet dataset are needed. For the training, it is
required to use the same dataset as the one used for training the teacher, therefore it is needed to
use a triplet dataset based on the train split of the VGGFace2 dataset. For testing, every dataset
can be used, but LFW is suggested since the results of the teacher are known for it.

The available models for distillation are MobileNetV3 Large and Small. To use either one or the
other one should set the `<model_name>` argument as `mobilenet_v3_large` or `mobilenet_v3_small`.

```
python -m src models distill <model_name> --train-set=<ID> --test-set=<ID> -e <NUM> -t <VAL> [--lr=<VAL>] [--no-lr-scheduler] [-b SIZE] [-w NUM]
```

### Test a network
To test the results of a neural network on a triplet dataset, it is possible to run the following
command.
```
python -m src models test (--student=<ID> | --teacher) --set=<ID> [--measure=<VAL>] [-b SIZE] [-w NUM]
```

Note that two accuracy measures are available: match accuracy and class accuracy. Match accuracy
evaluates the probability to correctly classify a face against two others, one of the same class and
the other of another class. On the other hand, class accuracy evaluates the likelihood to correctly
classify a face as its real identity.

Note that, for the time being, class accuracy is only available for triplet datasets built on the
train split of the VGGFace2 dataset and it is used only for insight purposes in development and
shouldn't be considered for evaluating the quality of a model.


# References
1. Moschoglou, S., Papaioannou, A., Sagonas, C., Deng, J., Kotsia, I., & Zafeiriou, S. (2017). AgeDB: The First Manually Collected, In-the-Wild Age Database. In IEEE Computer Society Conference on Computer Vision and Pattern Recognition Workshops (Vols. 2017-July). https://doi.org/10.1109/CVPRW.2017.250
2. Parkhi, O. M., Vedaldi, A., & Zisserman, A. (2015). Deep Face Recognition. British Machine Vision Association. https://doi.org/10.5244/c.29.41
3. Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. http://arxiv.org/abs/1503.02531
4. F. Schroff, D. Kalenichenko, J. Philbin. FaceNet: A Unified Embedding for Face Recognition and Clustering, arXiv:1503.03832, 2015. https://arxiv.org/pdf/1503.03832
5. Huang, G. B., Mattar, M., Berg, T., Labeled, E. L., Images, R., & Learned-miller, E. (2008). Labeled Faces in the Wild : A Database for Studying Face Recognition in Unconstrained Environments. Workshop on Faces in’Real-Life’Images: Detection, Alignment, and Recognition. https://hal.inria.fr/file/index/docid/321923/filename/Huang_long_eccv2008-lfw.pdf
6. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 4510–4520. https://doi.org/10.1109/CVPR.2018.00474
7. Howard, A., Sandler, M., Chen, B., Wang, W., Chen, L. C., Tan, M., Chu, G., Vasudevan, V., Zhu, Y., Pang, R., Le, Q., & Adam, H. (2019). Searching for mobileNetV3. Proceedings of the IEEE International Conference on Computer Vision, 2019-Octob, 1314–1324. https://doi.org/10.1109/ICCV.2019.00140
6. Q. Cao, L. Shen, W. Xie, O. M. Parkhi, A. Zisserman. VGGFace2: A dataset for recognising face across pose and age, International Conference on Automatic Face and Gesture Recognition, 2018. http://www.robots.ox.ac.uk/~vgg/publications/2018/Cao18/cao18.pdf 
7. Timesler's FaceNet implementation in PyTorch. https://github.com/timesler/facenet-pytorch
