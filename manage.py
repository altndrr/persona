"""
manage

Usage:
    manage env export
    manage -h | --help

Options:
    -h --help                   Show this screen.
"""

import os

from docopt import docopt


def main():
    """Project manager."""
    options = docopt(__doc__)

    # Dynamically match the user command with a function.
    for key, value in options.items():
        if key in globals() and value:
            command = globals()[key]
            command(options)


def env(options):
    """The env command."""
    if options['export']:
        print('Exporting conda packages to `environment.yml`...')

        # Get the name of the search utility given the os.
        utility = 'findstr' if os.name == 'nt' else 'grep'

        os.system(f'conda env export --from-history --no-builds | \
                    {utility} -v "prefix" > environment.yml')


if __name__ == '__main__':
    main()
