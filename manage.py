"""
manage

Usage:
    manage code (format | lint | test)
    manage -h | --help

Options:
    -h --help                   Show this screen.
"""

import os

from docopt import docopt


def main():
    """Main function of the manage script"""
    options = docopt(__doc__)

    # Dynamically match the user command with a function.
    for key, value in options.items():
        if key in globals() and value:
            command = globals()[key]
            command(options)


def code(options):
    """Code management command"""
    if options["format"]:
        os.system("black src && isort src")
    elif options["lint"]:
        os.system("pylint src")
    elif options["test"]:
        os.system(
            "pytest --cov=src --cov-report term-missing \
                   -n auto --timeout=60 src"
        )


if __name__ == "__main__":
    main()
