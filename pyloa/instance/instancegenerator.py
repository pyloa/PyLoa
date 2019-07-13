from collections.abc import Iterator
from abc import abstractmethod, ABCMeta
import logging


class InstanceGenerator(Iterator, metaclass=ABCMeta):
    """A generator to iteratively generate instance_number-many instances.

    Args:
        instance_number (int): number of instances to create

    Attributes:
        _instance_number (int): number of instances to create
    """
    def __init__(self, instance_number):
        self._instance_number = instance_number
        self._logger = logging.getLogger(__name__)

    def __len__(self):
        """int: returns the total number of instances the generator will generate before depletion"""
        return self._instance_number

    def __iter__(self):
        """creates an generator over self to generate instance

        Returns:
            generator: a generator function to iterate over and generate instances at runtime
        """
        return self.__next__()

    def __next__(self):
        """A generator function that generates len(self)-many instances

        Yields:
            instance: generated instance from a subclass implementation

        Raises:
            StopIteration: raised, when generator is depleted
        """
        for _ in range(len(self)):
            yield self._generate_instance()

    @abstractmethod
    def __str__(self):
        """str: string representation of instance generator"""
        raise NotImplementedError("Must be implemented in subclass.")

    @abstractmethod
    def _generate_instance(self):
        """Create and return a newly generated instance

        Returns:
            instance: returns instance for others to consume

        Raises:
            NotImplementedError: if abstract method is to be invoked
        """
        raise NotImplementedError("Must be implemented in subclass.")


