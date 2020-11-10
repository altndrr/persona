"""
main

Usage:
    main -h | --help
    main --version

Options:
    -h --help                   Show this screen.
    --version                   Show version.
"""

from docopt import docopt

from src import __version__


def main():
    """Main function of the module."""
    options = docopt(__doc__, version=__version__)

    # Dynamically match the user command with a function.
    for key, value in options.items():
        if key in globals() and value:
            command = globals()[key]
            command(options)
