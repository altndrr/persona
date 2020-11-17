"""The base command."""

from abc import ABC, abstractmethod


class Base(ABC):
    """A base command."""

    def __init__(self, options, *args, **kwargs):
        self.options = options
        self.args = args
        self.kwargs = kwargs

    @abstractmethod
    def run(self):
        """Run the command"""
