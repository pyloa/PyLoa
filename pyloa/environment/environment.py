import logging
from abc import abstractmethod, ABCMeta

import numpy as np


class Environment(metaclass=ABCMeta):
    """Abstract base class for Environments.

    Attributes:
        _logger (logging.Logger): logger instance for each subclassed instance
    """
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._logger.info("initializing {}".format(self.__class__.__name__))

    @abstractmethod
    def _get_action_dim(self):
        """Environment returns number of all actions.

        Returns:
            int: number of actions

        Raises:
            NotImplementedError: if abstract method is to be invoked
        """
        raise NotImplementedError("Must be implemented in subclass.")

    @abstractmethod
    def _get_state_dim(self):
        """Environment returns length of state vector.

        Returns:
            int: length of state vector

        Raises:
            NotImplementedError: if abstract method is to be invoked
        """
        raise NotImplementedError("Must be implemented in subclass.")

    @abstractmethod
    def get_state(self, vectorized_state=False, dtype=np.int32):
        """Environment returns state.

        Attributes:
            vectorized_state (bool): depicts if return type should be vectorized (np.ndarray) or non-vectorized (dict)

        Returns:
            np.ndarray or dict: state of environment

        Raises:
            NotImplementedError: if abstract method is to be invoked
        """
        raise NotImplementedError("Must be implemented in subclass.")

    @abstractmethod
    def calculate_state_complexity(self):
        """Environment calculates its state complexity.

        Returns:
            int: cardinality of state space

        Raises:
            NotImplementedError: if abstract method is to be invoked
        """
        raise NotImplementedError("Must be implemented in subclass.")

    @abstractmethod
    def step(self, action, vectorized_state=True):
        """Environment performs action and transition into next state.

        Attributes:
            action (int): action to take in current state
            vectorized_state (bool): depicts if return type should be vectorized (np.ndarray) or non-vectorized (dict)

        Returns:
            tuple: first element depicts the reward for action taken as a float number; second element is the state
                either as a np.ndarray or dict depending on vectorized_state (True for np.ndarray, False for dict)

        Raises:
            NotImplementedError: if abstract method is to be invoked
        """
        raise NotImplementedError("Must be implemented in subclass.")

    @abstractmethod
    def reset(self):
        """Environment resets and sets its internal structures to default values.

        Raises:
            NotImplementedError: if abstract method is to be invoked
        """
        raise NotImplementedError("Must be implemented in subclass.")

    @abstractmethod
    def _get_instance(self):
        raise NotImplementedError("Must be implemented in subclass.")

    @abstractmethod
    def _set_instance(self, value):
        raise NotImplementedError("Must be implemented in subclass.")

    instance = property(fset=_set_instance, fget=_get_instance, fdel=None)


class PagingEnvironment(Environment):
    """Abstract base class for pyloa.agent.PagingAgent agents to play on.

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
         _sequence_size (int): number of pages per sequence
         _cache_size (int): number of pages in cache
         _num_pages (int): number of different pages
         _num_steps (int): counter for current steps taken by agent so far
         _sequence (np.ndarray, shape=(_sequence_size,)): sequence of pages
         _cache (np.ndarray, shape=(_cache_sitze,), dtype=np.int32): pages in cache
         _time_in_cache (np.ndarray, shape=(_cache_sitze,), dtype=np.int32): time since page at index is in cache
         _histogram (np.ndarray, shape=(_cache_size,), dtype=np.int32): number of total requests for page in cache
         _time_since_touched (np.ndarray, shape=(_num_pages,), dtype=np.int32): time since page at index was requested
         _include_request (bool): depicts whether or not request (next page) is included within state
         _include_pagefault (bool): depicts whether or not is_pagefault is included within state
         _include_cache (bool): depicts whether or not _cache is included within state
         _include_time_in_cache (bool): depicts whether or not _time_im_cache is included within state
         _include_time_since_touched (bool): depicts whether or not _time_since_touched is included within state
         _include_histogram (bool): depicts whether or not _histogram is included within state

    Raises:
        NotImplementedError: if abstract class is to be instanced
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
        super().__init__()
        # params
        self._sequence_size = sequence_size
        self._cache_size = cache_size
        self._num_pages = num_pages
        self._num_steps = 0
        # structures
        self._sequence = np.zeros(self._sequence_size, dtype=np.int32)
        self._cache = np.zeros(self._cache_size, dtype=np.int32)
        self._time_in_cache = np.zeros(self._cache_size, dtype=np.int32)
        self._histogram = np.zeros(self._cache_size, dtype=np.int32)
        self._time_since_touched = np.zeros(self._cache_size, dtype=np.int32)
        self._default_cache = np.arange(start=1, stop=self._cache_size + 1)
        self._default_time_in_cache = np.flip(np.arange(start=1, stop=self._cache_size + 1))
        self._default_histogram = np.zeros(self._cache_size, dtype=np.int32)
        self._default_time_since_touched = np.flip(np.arange(start=1, stop=self._cache_size + 1))
        # state config
        self._include_request = include_request
        self._include_pagefault = include_pagefault
        self._include_cache = include_cache
        self._include_time_in_cache = include_time_in_cache
        self._include_time_since_touched = include_time_since_touched
        self._include_histogram = include_histogram

    def _get_action_dim(self):
        """int: number of actions"""
        return self._cache_size + 1

    def _get_cache(self):
        """np.ndarray: current cache of environment"""
        return self._cache

    def _get_cache_size(self):
        """int: max. number of pages in cache"""
        return self._cache_size

    def _get_histogram(self):
        """np.ndarray: current histogram of environment"""
        return self._histogram

    def _get_num_pages(self):
        """int: number of different pages (excludes '0' as placeholder for default page)"""
        return self._num_pages

    def _get_num_steps(self):
        """int: current number of actions/steps played on _sequence"""
        return self._num_steps

    def _get_request(self):
        """int: current page that is requested and has to be placed in cache"""
        return self._sequence[self.num_steps]

    def _get_time_in_cache(self):
        """np.ndarray: current _time_in_cache of environment"""
        return self._time_in_cache

    def _get_time_since_touched(self):
        """np.ndarray: current _time_since_touched of environment"""
        return self._time_since_touched

    def _get_instance(self):
        """np.ndarray: current sequence that agent plays on"""
        return self._sequence

    def _get_state_dim(self):
        """int: length of state vector"""
        return len(self.get_state(vectorized_state=True))

    def _get_used_indices(self):
        """np.ndarray: current indices in _cache that are beiung used (non-placeholder/default value)"""
        return np.where(self._cache != 0)[0]

    def _is_terminated(self):
        """bool: True, if environment is terminated (all requests for _sequence evaluated). False otherwise."""
        return self.num_steps == len(self._sequence) - 1

    def _is_pagefault(self):
        """bool: True, if requested page is not in _cache (page fault), else False."""
        return self.request not in self.cache

    def _set_instance(self, value):
        """Overrides currently stored _sequence with value without invoking reset()

        Args:
            value (np.ndarray): new _sequence of requests to store and play on

        Raises:
            AssertionError: if length of value does not equal _sequence_size or if value is not of type np.ndarray
        """
        assert isinstance(value, np.ndarray), "value must be of type='{}' but received type='{}' instead."\
            .format(np.ndarray, type(value))
        assert len(value) == self._sequence_size, "value must be of length='{}' but received length='{}' instead."\
            .format(self._sequence_size, len(value))
        self._sequence = value

    @abstractmethod
    def _calculate_cache_state_complexity(self):
        raise NotImplementedError("Must be implemented in subclass.")

    def _calculate_pagefault_state_complexity(self):
        """Calculates and returns the state complexity of page fault states, if page fault is included within
        environment's state representation; otherwise returns 1.

        Returns:
            int: 2, if page fault is included, else 1
        """
        return 2 if self._include_pagefault else 1

    def _calculate_request_state_complexity(self):
        """Calculates and returns the state complexity of request states, if request is included within environment's
        state representation; otherwise returns 1.

        Returns:
            int: _num_pages if request is included, else 1
        """
        return self.num_pages if self._include_request else 1

    @abstractmethod
    def _calculate_time_in_cache_state_complexity(self):
        raise NotImplementedError("Must be implemented in subclass.")

    @abstractmethod
    def _calculate_time_since_touched_state_complexity(self):
        raise NotImplementedError("Must be implemented in subclass.")

    @abstractmethod
    def _calculate_histogram_state_complexity(self):
        raise NotImplementedError("Must be implemented in subclass.")

    def calculate_state_complexity(self):
        """Calculates and returns the cardinality of state space.

        Returns:
            int: cardinality of state space
        """
        return self._calculate_request_state_complexity() * \
            self._calculate_pagefault_state_complexity() * \
            self._calculate_cache_state_complexity() * \
            self._calculate_time_in_cache_state_complexity() * \
            self._calculate_time_since_touched_state_complexity() * \
            self._calculate_histogram_state_complexity()

    def get_state(self, vectorized_state=False, vectorized_dtype=np.int32):
        """Returns the state of environment. The state is either vectorized (returns np.ndarray) or not (returns dict).
        The state comprises copies of each variable that has been defined to be included during environment creation.

        Args:
            vectorized_state: defines return type. A vectorized_state will be of type np.ndarray, otherwise dict
            vectorized_dtype: dtype of np.darray for vectorized_state

        Returns:
            np.ndarray or dict: state depicting the current observation (changing the returned state won't change the
                internal structures in environment)
        """
        if vectorized_state:
            _state = np.asarray([], dtype=vectorized_dtype)
            if self._include_request:
                _state = np.append(_state, np.asarray(self.request, dtype=vectorized_dtype))
            if self._include_pagefault:
                _state = np.append(_state, np.asarray([1 if self.is_pagefault else 0], dtype=vectorized_dtype))
            if self._include_cache:
                _state = np.append(_state, self.cache)
            if self._include_time_in_cache:
                _state = np.append(_state, self.time_in_cache)
            if self._include_time_since_touched:
                _state = np.append(_state, self.time_since_touched)
            if self._include_histogram:
                _state = np.append(_state, self.histogram)
        else:
            _state = dict()
            _state['current_page'] = self.request
            _state['is_pagefault'] = self.is_pagefault
            _state['cache'] = np.copy(self.cache)
            _state['time_in_cache'] = np.copy(self.time_in_cache)
            _state['time_since_touched'] = np.copy(self.time_since_touched)
            _state['histogram'] = np.copy(self.histogram)
            _state['num_steps'] = self.num_steps
        return _state

    def reset(self, cache=None, time_in_cache=None, time_since_touched=None, histogram=None):
        """Resets the environment to its initial state.

        Args:
            cache (collections.abc.Sequence): initial cache after reset
            time_in_cache (collections.abc.Sequence): initial time_in_cache after reset
            time_since_touched (collections.abc.Sequence): initial time_since_touched after reset
            histogram (collections.abc.Sequence): initial histogram after reset
        """
        self._num_steps = 0
        self._cache[:] = self._default_cache if cache is None else cache
        self._time_in_cache[:] = self._default_time_in_cache if time_in_cache is None else time_in_cache
        self._time_since_touched[:] = self._default_time_since_touched if time_since_touched is None else \
            time_since_touched
        self._histogram[:] = self._default_histogram if histogram is None else histogram

    def step(self, action, vectorized_state=True):
        """Environment computes the step for action, updates all internal structures and returns the next state and cost
        as a tuple.

        Args:
            action (int): depicts the action (index) that the agent wants to flush (or 0 for no-action)
            vectorized_state (bool): depicts the return type of the observation after step/action is taken

        Returns:
            tuple: first entry is the next state, second entry are the costs affiliated with last action in last state

        Raises:
            AssertionError: if action is not in interval [0, cache_size] and therefore invalid
        """
        # check bounds (0 means 'no action',  interval [1, cache_size] represents indices (offset by one))
        assert 0 <= action <= self._cache_size, "action must be in interval [0, {}] but received action='{}' instead"\
            .format(self._cache_size, action)
        used_indices = self.used_indices

        # no action (no flush, current_page expected by agent to exist in cache)
        if action == 0:
            # agent is right, page exists in cache -> update structures
            if not self.is_pagefault:
                idx = np.where(self._cache == self.request)[0][0]
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
                idx = np.where(self._cache == self.request)[0][0]
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
                self._cache[idx] = self.request
                self._time_since_touched[used_indices] += 1
                self._time_since_touched[idx] = 0
                self._time_in_cache[used_indices] += 1
                self._time_in_cache[idx] = 1
                self._histogram[idx] = 1
        # inc step
        self._num_steps += 1
        return self.get_state(vectorized_state=vectorized_state), cost

    action_dim = property(fset=None, fget=_get_action_dim, fdel=None)
    cache = property(fset=None, fget=_get_cache, fdel=None)
    cache_size = property(fset=None, fget=_get_cache_size, fdel=None)
    histogram = property(fset=None, fget=_get_histogram, fdel=None)
    instance = property(fset=_set_instance, fget=_get_instance, fdel=None)
    is_pagefault = property( fset=None, fget=_is_pagefault, fdel=None)
    is_terminated = property(fset=None, fget=_is_terminated, fdel=None)
    num_pages = property(fset=None, fget=_get_num_pages, fdel=None)
    num_steps = property(fset=None, fget=_get_num_steps, fdel=None)
    request = property(fset=None, fget=_get_request, fdel=None)
    state_dim = property(fset=None, fget=_get_state_dim, fdel=None)
    used_indices = property(fset=None, fget=_get_used_indices, fdel=None)
    time_in_cache = property(fset=None, fget=_get_time_in_cache, fdel=None)
    time_since_touched = property(fset=None, fget=_get_time_since_touched, fdel=None)
