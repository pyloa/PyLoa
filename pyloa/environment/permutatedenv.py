import numpy as np

from .environment import PagingEnvironment


# Todo @schoenitz documentation
class PermutatedPagingEnvironment(PagingEnvironment):
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
                 dtype=np.int32,
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
            dtype=dtype,
            **kwargs
        )

        _cache_perm = {i + 1: i + 1 for i in range(self._cache_size)}
        _val_perm = {i: i for i in range(self._num_pages + 1)}
        self._permutation = {**_cache_perm, **_val_perm}

    def _update_permutation(self, pos, val):
        """We apply a transposition on the permutation itself"""
        if pos not in self._permutation.keys():
            raise ValueError(pos, "(as position) is not a valid key in the permutation:", self._permutation)
        if val not in self._permutation.keys():
            raise ValueError(val, "(as value) is not a valid key in the permutation:", self._permutation)
        self._permutation[pos], self._permutation[val] = self._permutation[val], self._permutation[pos]

    def _get_val_after_permutation(self, val):
        if val not in self._permutation.keys():
            raise ValueError(val, "is not a valid key in the permutation:", self._permutation)
        return self._permutation[val]

    def _get_histogram(self):
        return self._histogram

    def _get_time_in_cache(self):
        return self._time_in_cache

    def _get_time_since_touched(self):
        return self._time_since_touched

    def _calculate_cache_state_complexity(self):
        if self._include_cache:
            return 2 ** self._cache_size
        else:
            return 1

    def _calculate_time_in_cache_state_complexity(self):
        pass

    def _calculate_time_since_touched_state_complexity(self):
        pass

    def _calculate_histogram_state_complexity(self):
        pass

    def _is_pagefault(self):
        return self._get_val_after_permutation(self.request) not in self.cache

    def step(self, action, vectorized_state=True):
        # check bounds (0 means 'no action',  interval [1, cache_size] represents indices (offset by one))
        if not 0 <= action <= self._cache_size:
            raise ValueError("Out ouf bounds")
        used_indices = self.used_indices

        permutated_request = self._get_val_after_permutation(self.request)

        # no action (no flush, current_page expected by agent to exist in cache)
        if action == 0:
            # agent is right, page exists in cache -> update structures
            if not self.is_pagefault:
                idx = permutated_request - 1
                self._time_since_touched[used_indices] += 1
                self._time_since_touched[idx] = 0
                self._time_in_cache[used_indices] += 1
                self._histogram[idx] += 1
                cost = 0  # no costs, since agent correctly identified that current_page is in cache
            # agent is incorrect, page doesn't exist in cache
            else:
                self._time_since_touched[used_indices] += 1
                self._time_in_cache[used_indices] += 1
                cost = 1000  # high costs, since agent falsely expected current_page to be in cache
        # agent choose action to flush index to place current_page
        else:
            # agent tries to flush an index, even though current_page is already in cache
            if not self.is_pagefault:
                idx = np.where(self._cache == permutated_request)[0][0]
                self._time_since_touched[used_indices] += 1
                self._time_since_touched[idx] = 0
                self._time_in_cache[used_indices] += 1
                self._histogram[idx] += 1
                cost = 500
            # agent is right, page doesn't exist in cache -> flush selected action index
            else:
                idx = action - 1
                # agent decides to place current_page in empty cache slot -> no flush
                if 0 in self._cache and self._cache[idx] == 0:
                    cost = 0
                # agent decide to place current_page in a non-empty index, even though there is free space
                elif 0 in self._cache and self._cache[idx] != 0:
                    cost = 250
                # normal flush
                else:
                    cost = 10
                # self._cache[idx] = self.request
                self._cache[idx] = idx + 1
                self._update_permutation(idx + 1, permutated_request)
                self._time_since_touched[used_indices] += 1
                self._time_since_touched[idx] = 0
                self._time_in_cache[used_indices] += 1
                self._time_in_cache[idx] = 1
                self._histogram[idx] = 1
        # inc step
        self._num_steps += 1
        return self.get_state(vectorized_state=vectorized_state), cost

    histogram = property(fset=None, fget=_get_histogram, fdel=None)
    time_in_cache = property(fset=None, fget=_get_time_in_cache, fdel=None)
    time_since_touched = property(fset=None, fget=_get_time_since_touched, fdel=None)
    is_pagefault = property(fset=None, fget=_is_pagefault, fdel=None)
