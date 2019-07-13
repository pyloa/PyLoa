import math

import numpy as np
from scipy.stats import rankdata
from scipy.special import comb

from .environment import PagingEnvironment


class NormalizedPagingEnvironment(PagingEnvironment):
    """Normalized Paging Environment for pyloa.agent.PagingAgent agents to play on.

    Normalized Paging Environment ranks its internal structures (_histogram, _time_in_cache, _time_since_touched) when
    they are being requested via their getters. _rank_method depicts how ranks are assigned to equal values ('average',
    'min', 'max', 'dense', 'ordinal').

    Args:
        sequence_size (int): number of pages per sequence
        cache_size (int): number of pages in cache
        num_pages (int): number of different pages
        include_request (bool): depicts whether or not request (next page) is included within state
        include_pagefault (bool): depicts whether or not is_pagefault is included within state
        include_cache (bool): depicts whether or not _cache is included within state
        include_time_in_cache (bool): depicts whether or not _time_im_cache is included within state
        include_time_since_touched (bool): depicts whether or not _time_since_touched is included within state
        include_histogram (bool): depicts whether or not _histogram is included within state

    Attributes:
        _rank_method (str): depicts how ranks are assigned to equal values, check scipy.stats.rankdata for reference
        _cache_time_ranks (np.ndarray, shape=(_cache_size,), dtype=np.int32): ranking of _time_in_cache
        _touched_time_ranks (np.ndarray, shape=(_cache_size,), dtype=np.int32): ranking of _time_since_touched
        _histogram_ranks (np.ndarray, shape=(_cache_size,), dtype=np.int32): ranking of _histogram
    """
    def __init__(self,
                 sequence_size,
                 cache_size,
                 num_pages,
                 include_request=False,
                 include_pagefault=False,
                 include_cache=False,
                 include_time_in_cache=False,
                 include_time_since_touched=False,
                 include_histogram=False,
                 rank_method='dense',
                 **kwargs):
        super().__init__(
            sequence_size=sequence_size,
            cache_size=cache_size,
            num_pages=num_pages,
            include_request=include_request,
            include_pagefault=include_pagefault,
            include_cache=include_cache,
            include_time_in_cache=include_time_in_cache,
            include_time_since_touched=include_time_since_touched,
            include_histogram=include_histogram,
            **kwargs
        )
        # params
        self._rank_method = rank_method
        # structures
        self._cache_time_ranks = np.zeros(self._cache_size, dtype=np.int32)
        self._touched_time_ranks = np.zeros(self._cache_size, dtype=np.int32)
        self._histogram_ranks = np.zeros(self._cache_size, dtype=np.int32)

    def _calculate_cache_state_complexity(self):
        """Calculates and returns the state complexity of cache states, if cache is included within environment's
        state representation; otherwise returns 1.

            #states = num_pages! * sum_{i=0}^{cache_size+1} binom{num_pages}{cache_size} / (num_pages - cache_size + i)!

        Returns:
            int: cardinality of cache-state space if included, else 1
        """
        if self._include_cache:
            _fac_num_pages = math.factorial(self.num_pages)
            res = 0
            for i in range(0, self._cache_size+1):
                _comb = comb(N=self.num_pages, k=self._cache_size, exact=True)
                # Todo: is this correct? is _comb really independent from i??? -> move it out of the loop at least?
                res += (_fac_num_pages * _comb) // math.factorial(self.num_pages - self._cache_size + i)
        else:
            res = 1
        return res

    # Todo: define ranked time complexity
    def _calculate_time_in_cache_state_complexity(self):
        raise NotImplementedError()

    # Todo: define ranked time since touched complexity
    def _calculate_time_since_touched_state_complexity(self):
        raise NotImplementedError()

    # Todo: define ranked histogram complexity
    def _calculate_histogram_state_complexity(self):
        raise NotImplementedError()

    def _get_time_in_cache(self, maximum_first=True):
        """Calculates and returns the rank of values in _time_in_cache. Placeholder values in _cache have rank 0.

        Args:
            maximum_first (bool): if True, highest value will have rank 1; else highest value will have last rank.

        Returns:
            np.ndarray: rank for each index in _time_in_cache
        """
        return self._rank_array(self._time_in_cache, self._cache_time_ranks, maximum_first)

    def _get_time_since_touched(self, maximum_first=True):
        """Calculates and returns the rank of values in _time_since_touched. Placeholder values in _cache have rank 0.

        Args:
            maximum_first (bool): if True, highest value will have rank 1; else highest value will have last rank.

        Returns:
            np.ndarray: rank for each index in _time_since_touched
        """
        return self._rank_array(self._time_since_touched, self._touched_time_ranks, maximum_first)

    def _get_histogram(self, maximum_first=True):
        """Calculates and returns the rank of values in _histogram. Placeholder values in _cache have rank 0.

        Args:
            maximum_first (bool): if True, highest value will have rank 1; else highest value will have last rank.

        Returns:
            np.ndarray: rank for each index in _histogram
        """
        return self._rank_array(self._histogram, self._histogram_ranks, maximum_first)

    def _rank_array(self, source_array, target_array, maximum_first=True):
        """Calculates the rank of values in source_array and places their rank in target_array. Placeholder values in
        _cache have rank 0.

        Args:
            maximum_first (bool): if True, highest value will have rank 1; else highest value will have last rank.

        Returns:
            np.ndarray: rank for each index in source_array
        """
        target_array.fill(0)
        _used_indices = self.used_indices
        _used_source = source_array[_used_indices]
        _used_source = np.iinfo(np.int32).max - _used_source if maximum_first else _used_source
        _ranked_used_source = rankdata(_used_source, method=self._rank_method)
        target_array[_used_indices] = _ranked_used_source
        return target_array

    time_in_cache = property(fset=None, fget=_get_time_in_cache, fdel=None)
    time_since_touched = property(fset=None, fget=_get_time_since_touched, fdel=None)
    histogram = property(fset=None, fget=_get_histogram, fdel=None)
