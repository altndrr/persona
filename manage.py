"""
manage

Usage:
    manage code (-f | --format | -l | --lint)...
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


if __name__ == "__main__":
    main()
