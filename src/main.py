"""
main

Usage:
    main models distill <model_name> --train-set=[<ID> | vggface2] --test-set=[<ID> | lfw] \
-e <NUM> -t <VAL> [--lr=<VAL>] [--no-lr-scheduler] [-b SIZE] [-w NUM]
    main models list
    main models test (--student=<ID> | --teacher) --set=[<ID> | lfw] [--measure=<VAL>] \
[-b SIZE] [-w NUM]
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
    --set=[<ID> | lfw | vggface2]   ID of a triplet (either for training or for testing) \
or dataset name.
    --split=<VAL>                   Split of the dataset, either train or test.
    --student=<ID>                  ID of the student network.
    --train-set=[<ID> | vggface2]   ID of the training set or dataset name.
    --test-set=[<ID> | lfw]         ID of the testing set or dataset name.
    -h --help                       Show this screen.
    --version                       Show version.
"""
from inspect import getmembers, isclass

from docopt import docopt

import src.utils.commands
from src import __version__


def main():
    """Main function of the module"""
    options = docopt(__doc__, version=__version__)

    options = parse_options(options)

    # Try to dynamically match the command the user is trying to run
    # with a pre-defined command class we've already created.
    for key, value in options.items():
        if hasattr(src.utils.commands, key) and value:
            module = getattr(src.utils.commands, key)
            commands = [
                namespace
                for name, namespace in getmembers(module, isclass)
                if "commands" in str(namespace) and name != "Base"
            ]
            command = commands[0]
            command = command(options)
            command.run()


def parse_options(options):
    """
    Parse the options passed by the user.

    :param options: dictionary of options
    :return: parsed dictionary
    """
    options["--batch"] = int(options["--batch"])
    if options["--epochs"]:
        options["--epochs"] = int(options["--epochs"])
    options["--lr"] = float(options["--lr"])
    if options["--measure"] not in ["class", "match"]:
        raise ValueError(f'{options["--measure"]} is an invalid test measure')
    if options["--set"]:
        if options["--set"] != "lfw":
            options["--set"] = int(options["--set"])
    if options["--split"] not in [None, "test", "train"]:
        raise ValueError(f'{options["--measure"]} is an invalid test measure')
    if options["--student"]:
        options["--student"] = int(options["--student"])
    if options["--temperature"]:
        options["--temperature"] = int(options["--temperature"])
    if options["--test-set"]:
        if options["--test-set"] != "lfw":
            options["--test-set"] = int(options["--test-set"])
    if options["--train-set"]:
        if options["--train-set"] != "vggface2":
            options["--train-set"] = int(options["--train-set"])
    options["--workers"] = int(options["--workers"])
    if options["<dataset>"] not in [None, "agedb", "lfw", "vggface2"]:
        raise ValueError(f'{options["<dataset>"]} is an invalid dataset')
    if options["<model_name>"] not in [
        None,
        "mobilenet_v3_large",
        "mobilenet_v3_small",
    ]:
        raise ValueError(f'{options["<model_name>"]} is an invalid model name')
    if options["<num_triplets>"]:
        options["<num_triplets>"] = int(options["<num_triplets>"])

    return options
