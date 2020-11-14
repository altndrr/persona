"""
manage

Usage:
    manage code (-f | --format | -l | --lint | -t | --test)...
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
    if options["--format"] or options["-f"]:
        os.system("black src && isort src")

    if options["--lint"] or options["-l"]:
        os.system("pylint src")

    if options["--test"] or options["-t"]:
        os.system(
            "pytest --cov=src --cov-report term-missing \
                   -n auto --timeout=180 src"
        )


if __name__ == "__main__":
    main()
