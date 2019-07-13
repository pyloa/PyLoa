import numpy as np

from .instancegenerator import InstanceGenerator


class SequenceGenerator(InstanceGenerator):
    """A generator to iteratively generate sequence_number-many sequences of size sequence_size.

    Args:
        sequence_size (int): target size of each sequence
        sequence_number (int): number of sequences to generate till generator depletes
        min_page (int): minimum element within each generated sequence
        max_page (int): maximum element within each generated sequence
        dtype (type): type of each element within sequence

    Attributes:
        _sequence_size (int): target size of each sequence
        _min_page (int): minimum element within each generated sequence
        _max_page (int): maximum element within each generated sequence
        _num_pages (int): number of different pages that can occur in each generated sequence
        _dtype (type): type of each element within sequence
        _pages (np.ndarray, np.shape(_num_pages,), dtype=_dtype): lookup list for each page that can occur

    Raises:
        AssertionError: if sequence_size, sequence_number or min_page does not exceed 0 or if max_page exceeds min_page
        NotImplementedError: if abstract class is to be instanced
    """
    def __init__(self,
                 sequence_size,
                 sequence_number,
                 min_page,
                 max_page,
                 dtype=np.int32):
        assert sequence_size > 0, "sequence_size must exceed 0"
        assert sequence_number > 0, "sequence_number must exceed 0"
        assert min_page > 0, "min_page must exceed 0"
        assert max_page > min_page, "max_page must exceed min_page"
        super(SequenceGenerator, self).__init__(instance_number=sequence_number)
        self._sequence_size = sequence_size
        self._min_page = min_page
        self._max_page = max_page
        self._num_pages = max_page - min_page + 1
        self._dtype = dtype
        self._pages = np.arange(self._num_pages, dtype=self._dtype) + self._min_page

    def _get_pages(self):
        """np.ndarray: ascending array of pages self can populate sequences with"""
        return self._pages

    def _get_sequence_size(self):
        """int: number of pages per sequence"""
        return self._sequence_size

    pages = property(fset=None, fget=_get_pages, fdel=None)
    sequence_size = property(fset=None, fget=_get_sequence_size, fdel=None)
