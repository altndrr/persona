"""
manage

Usage:
    manage -h | --help

Options:
    -h --help                   Show this screen.
"""

from docopt import docopt


def main():
    """Project manager."""
    options = docopt(__doc__)

    # Dynamically match the user command with a function.
    for key, value in options.items():
        if key in globals() and value:
            command = globals()[key]
            command(options)


if __name__ == '__main__':
    main()
