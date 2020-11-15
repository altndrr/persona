"""
main

Usage:
    main generate triplets -n <number> -p <processes> <dataset> [<split>]
    main -h | --help
    main --version

Options:
    -h --help                   Show this screen.
    --version                   Show version.
"""
import os

from docopt import docopt

from src import __version__
from src.data.raw import AgeDB, VGGFace2
from src.features.generate import triplets


def main():
    """Main function of the module"""
    options = docopt(__doc__, version=__version__)

    if "generate" in options:
        generate(options)


def generate(options):
    """Generate command"""

    if "triplets" in options:
        try:
            options["<number>"] = int(options["<number>"])
            options["<processes>"] = int(options["<processes>"])
        except ValueError as value_error:
            raise value_error

        if options["<processes>"] == 0:
            options["<processes>"] = os.cpu_count()

        if options["<dataset>"] == "agedb":
            dataset = AgeDB()
        elif options["<dataset>"] == "vggface2" and options["<split>"]:
            dataset = VGGFace2(options["<split>"])
        else:
            raise ValueError()

        triplets(dataset, options["<number>"], options["<processes>"])
