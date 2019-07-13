import math

from scipy.special import comb

from pyloa.utils.math import s2
from .environment import PagingEnvironment


class DefaultPagingEnvironment(PagingEnvironment):
    """Default Paging Environment for pyloa.agent.PagingAgent agents to play on.

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
    """
    def __init__(self,
                 sequence_size,
                 cache_size,
                 num_pages,
                 include_request=True,
                 include_pagefault=False,
                 include_cache=False,
                 include_time_in_cache=False,
                 include_time_since_touched=False,
                 include_histogram=False,
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

    def _calculate_time_in_cache_state_complexity(self):
        """Calculates and returns the state complexity of time_in_cache states, if time_in_cache is included within
        environment's state representation; otherwise returns 1.

            #states = _sequence_size ** _cache_size

        Returns:
            int: cardinality of time_in_cache-state space if included, else 1
        """
        if self._include_time_in_cache:
            res = self._sequence_size ** self._cache_size
            # Todo: this is not correct (time_in_cache has unique values)
        else:
            res = 1
        return res

    def _calculate_time_since_touched_state_complexity(self):
        """Calculates and returns the state complexity of time_since_touched states, if time_since_touched is included
        within environment's state representation; otherwise returns 1.

            # states = sum_{c=1}^{cache_size+1} binom{cache_size}{c} * sum_{k=1}^{sequence_size+1} S(k, c)

        whereas S denotes the Stirling number of the second kind (pyloa.utils.s2).

        Returns:
            int: cardinality of time_since_touched-state space if included, else 1
        """
        if self._include_time_since_touched:
            res = 0
            for c in range(1, self._cache_size+1):
                _comb = comb(N=self._cache_size, k=c, exact=True)
                _sum = 0
                for k in range(1, self._sequence_size+1):
                    _sum += s2(k, c)
                res += _sum * _comb
        else:
            res = 1
        return res

    def _calculate_histogram_state_complexity(self):
        """Calculates and returns the state complexity of histogram states, if histogram is included within
        environment's state representation; otherwise returns 1.

            # states = sum_{c=1}^{cache_size+1} binom{cache_size}{c} * sum_{k=1}^{sequence_size+1} S(k, c)

        whereas S denotes the Stirling number of the second kind (pyloa.utils.s2).

        Returns:
            int: cardinality of time_since_touched-state space if included, else 1
        """
        if self._include_histogram:
            res = 0
            for c in range(1, self._cache_size+1):
                _comb = comb(N=self._cache_size, k=c, exact=True)
                _sum = 0
                for k in range(1, self._sequence_size+1):
                    _sum += s2(k, c)
                res += _sum * _comb
        else:
            res = 1
        return res
