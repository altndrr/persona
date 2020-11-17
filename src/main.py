"""
main

Usage:
    main models distill <model_name> --train-set <train_set_id> --test-set <test_set_id> \
-e <epochs> -t <temperature> [--decay <decay_type>] [--no-lr-scheduler]
    main models test (--id <model_id> | teacher) --test-set <test_set_id>
    main triplets list
    main triplets make -n <number> -p <processes> <dataset> [<split>]
    main -h | --help
    main --version

Options:
    -h --help                   Show this screen.
    --version                   Show version.
"""
from inspect import getmembers, isclass

from docopt import docopt

import src.utils.commands
from src import __version__


def main():
    """Main function of the module"""
    options = docopt(__doc__, version=__version__)

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
