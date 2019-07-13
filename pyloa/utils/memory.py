import numpy as np
from abc import abstractmethod
from collections.abc import Iterable, Iterator

from .sumtree import SumTree


class Memory(Iterator):
    """Abstract Replay Memory base class for Agents to store experiences in and draw from.

    Each Memory implementation must be agnostic with regards to its experiences; hence, return types of Memory are
    referred to as type(experience).

    Args:
        memory_size (int): number of experiences to store

    Attributes:
        _memory_size(int): number of experiences to store

    Raises:
        NotImplementedError: if abstract method is to be invoked
    """
    def __init__(self, memory_size):
        self._memory_size = memory_size
        if type(self) == Memory:
            raise NotImplementedError("{} must be subclassed.".format(self.__class__.__name__))

    @abstractmethod
    def __getitem__(self, item):
        """get experiences(s) indexed with item

        Args:
            item (number, slice, Iterable): a selector to index experience with in self

        Returns:
             type(experience): if isinstance(item, number)
             tuple of type(experience): if isinstance(item, (Iterable, slice))

        Raises:
            NotImplementedError: if abstract method is to be invoked
        """
        raise NotImplementedError("Must be implemented in subclass.")

    def __iter__(self):
        """generator: generator to iterate with over self"""
        return self.__next__()

    def __next__(self):
        """type(experience): iterate over len, yield experience at index"""
        for index in range(len(self)):
            yield self[index]

    @abstractmethod
    def __len__(self):
        """int: number of currently stored experiences"""
        raise NotImplementedError("Must be implemented in subclass.")

    @abstractmethod
    def clear(self):
        """clears Memory and resets all necessary data structures"""
        raise NotImplementedError("Must be implemented in subclass.")

    @abstractmethod
    def sample(self, sample_size):
        """Draw experiences of size sample_size.

        Args:
            sample_size (int): sample size to be drawn

        Returns:
            tuple of type(experience): tuple of drawn experiences with length sample_size
        """
        raise NotImplementedError("Must be implemented in subclass.")

    @abstractmethod
    def store(self, experience):
        """add experience to Memory"""
        raise NotImplementedError("Must be implemented in subclass.")


class ReplayMemory(Memory):
    """A light-weight ring buffer for containing experiences. Experiences are drawn uniformly.

    Args:
        memory_size (int): number of experiences to store
        seed (int or None): seed for random choice

    Attributes:
        _data (list): internal container for experiences to be stored
        _data_cursor (int): cursor for ring-buffer implementation
        _rnd_state (np.random.RandomState): random number generator for self with own local seed
        _seed (int or None): seed _rnd_state was initialized with
    """
    def __init__(self, memory_size, seed=None):
        super().__init__(memory_size=memory_size)
        self._data = list()
        self._data_cursor = 0
        self._seed = seed
        self._rnd_state = np.random.RandomState(seed=seed)

    def __setitem__(self, index, value):
        """set value/s at index/indices"""
        self._data[index] = value

    def __getitem__(self, item):
        """Retrieve experience(s) stored in ReplayMemory indexed with item.

        Args:
            item (number, slice, Iterable): a selector to index experience with in self

        Returns:
             type(experience): if isinstance(item, number)
             tuple of type(experience): if isinstance(item, (Iterable, slice))

        Raises:
            TypeError: item selection not defined with type of item
        """
        if isinstance(item, Iterable):
            return tuple(map(self._data.__getitem__, item))
        if isinstance(item, slice):
            return tuple(self._data[item])
        if isinstance(item, (int, float, np.number)):
            return self._data[item]
        raise TypeError("{} item is of unexpected type='{}': can't perform get operation"
                        .format(self.__class__.__name__, type(item)))

    def __len__(self):
        """int: len of self"""
        return len(self._data)

    def store(self, experience):
        """Store experience in Memory. If _memory_size is not reached yet, append experience; else store at _data_cursor
        and move _data_cursor.

        Args:
            experience (type(experience)): experience to store
        """
        if len(self) < self._memory_size:
            self._data.append(experience)
        else:
            self[self._data_cursor] = experience
            self._data_cursor = (self._data_cursor + 1) % self._memory_size

    def clear(self):
        """Clear memory."""
        del self._data[:]
        self._data_cursor = 0

    def sample(self, sample_size):
        """Uniformly draw sample of size sample_size from Memory.

        Args:
            sample_size (int): sample size to be drawn

        Returns:
            tuple of type(experience): tuple of drawn experiences with length sample_size
        """
        rnd_idx = self._rnd_state.randint(low=0, high=len(self), size=sample_size, dtype=np.int32)
        return self[rnd_idx]


class PrioritizedReplayMemory(Memory):
    """An implementation of Prioritized Replay Memory according to Schaul et. al.: https://arxiv.org/pdf/1511.05952.pdf

    Prioritizes important experiences to be picked more frequently based on td_errors from a QAgent. This implementation
    uses a binary indexed tree as an indexing structure to sample 'more valuable' experiences (commonly referred to as
    SumTree or Fenwick tree). For stability reasons td_errors have to be normalized via td_normalization to fall within
    [0.0, 1.0]. Priority updates within the index structure can result in small delta updates, which can not be
    propagated upwards throughout the entire tree due to floating point precision limitations; the SumTree
    implementation at hand checks for "failing" delta updates and auto-corrects "too small delta changes" to the next
    applicable machine epsilon. If the auto-correct exceeds a relative change of max_relative_change, a warning is
    logged.

    Args:
        memory_size (int): number of experiences to store
        alpha (float): defines how much prioritization is used (alpha = 0.0 correspond to the uniform case)
        beta (float): defines how much importance sampling is used (linearly increased for each global step)
        beta_delta (float): defines by how much beta is increased after each global step
        eps (float): smallest possible priority to clip 0.0
        td_normalization (number): value by which td_errors is divided to normalize td_error in [0.0, 1.0]
        max_relative_change (number): maximum relative change of SumTree's auto-correction for numerical stability
        tree_dtype (type): dtype for floating point precision used in SumTree
        seed (int or None): seed for random choice

    Attributes:
        _alpha (float): defines how much prioritization is used (alpha = 0.0 correspond to the uniform case)
        _beta (float): defines how much importance sampling is used (linearly increased for each global step)
        _beta_delta (float): defines by how much beta is increased after each global step
        _tree (utils.SumTree): a Fenwick tree instance that indexes _memory_size many experiences
        _eps (float): smallest possible priority to clip 0.0
        _td_normalization (number): value by which td_errors is divided to normalize td_error in [0.0, 1.0]
        _rnd_state (np.random.RandomState): random number generator for self with own local seed
        _seed (int or None): seed for random choice
    """
    def __init__(self,
                 memory_size,
                 alpha,
                 beta,
                 beta_delta,
                 eps,
                 td_normalization,
                 max_relative_change=1e3,
                 tree_dtype=np.float64,
                 seed=None,
                 **kwargs):
        super().__init__(memory_size=memory_size)
        self.alpha = alpha
        self.beta = beta
        self.beta_delta = beta_delta
        self._tree = SumTree(memory_size=self._memory_size, max_relative_change=max_relative_change, dtype=tree_dtype)
        self._eps = eps
        self._td_normalization = td_normalization
        self._seed = seed
        self._rnd_state = np.random.RandomState(seed=self._seed)

    def __len__(self):
        """int: number of experiences stored"""
        return self._tree.data_size

    def __getitem__(self, item):
        """Retrieve experience(s) stored in ReplayMemory indexed with item.

        Args:
            item (number, slice, Iterable): a selector to index experience with in _tree

        Returns:
             tuple of type(experience):
        """
        return self._tree[item]

    def clear(self):
        """Clear memory."""
        self._tree.clear()

    # noinspection PyMethodOverriding
    def sample(self, sample_size):
        """Draw sample of size sample_size from Memory based on priorities.

        Args:
            sample_size (int): sample size to be drawn

        Returns:
            tuple(tuple(int), tuple(float), list(type(experience))): First entry depicts tree index at which each
                experience was drawn from (reused for update after batch is trained on). Second entry depicts priority
                value of experience. Last entry is the experience itself.
        """
        _total_priority = self._tree.total_priority
        _alpha_priority = np.sum(np.power(self._tree.priorities, self.alpha))
        _segment_num = int(_total_priority // sample_size)
        _segment_values = [self._rnd_state.uniform(_segment_num*i, _segment_num*(i + 1)) for i in range(sample_size)]
        _tree_indices, _priorities, _experiences = zip(*self[_segment_values])
        _sampled_weights = [(self._memory_size * (p ** self.alpha) / _alpha_priority) ** (-self.beta) for p in _priorities]
        _max_sample_weight = max(_sampled_weights)
        _sampled_weights[:] = [s / _max_sample_weight for s in _sampled_weights]
        return _tree_indices, _sampled_weights, _experiences

    def store(self, experience):
        """Store experience in Memory. Each new experience is initialized with max_priority to ensure it's trained upon
        at least once.

        Args:
            experience type(experience): experience to be stored
        """
        _max_priority = self._tree.max_priority
        _max_priority = _max_priority if _max_priority != 0.0 else 1.0
        self._tree.add(priority=_max_priority, data_entry=experience)

    def update(self, tree_indices, td_errors):
        """Updates memory after last sample (batch) is processed.

        Args:
            tree_indices (tuple of int): tree index at which experience was drawn from prior sample() invocation
            td_errors (np.ndarray of float): td_error for each experience from prior sample() invocation
        """
        td_errors = td_errors / self._td_normalization              # normalize td_error to be in [0.0, 1.0]
        td_errors[td_errors == 0.0] += self._eps                    # clip to be in [self._eps, 1.0 + self._eps]
        td_errors = np.minimum(td_errors, 1.0)                      # clip to be in [self._eps, 1.0]
        for tree_index, td_error in zip(tree_indices, td_errors):   # update tree priorities
            self._tree.update(tree_index, td_error)

    def increment_beta(self):
        """increment beta by beta_delta clipped at 1.0"""
        self.beta = min(self.beta + self.beta_delta, 1.0)

    def _set_alpha(self, value):
        assert 0.0 <= value <= 1.0, "alpha must be in interval [0.0, 1.0] but received value='{}'instead".format(value)
        self._alpha = value

    def _set_beta(self, value):
        assert 0.0 <= value <= 1.0, "beta must be in interval [0.0, 1.0] but received value='{}'instead".format(value)
        self._beta = value

    def _set_beta_delta(self, value):
        assert 0.0 <= value <= 1.0, "beta_delta must be in interval [0.0, 1.0] but received value='{}'instead"\
            .format(value)
        self._beta_delta = value

    def _get_alpha(self):
        """float: defines how much prioritization is used (alpha = 0.0 correspond to the uniform case)"""
        return self._alpha

    def _get_beta(self):
        """float: defines how much importance sampling is used (linearly increased for each global step)"""
        return self._beta

    def _get_beta_delta(self):
        """float: defines by how much beta is increased after each global step"""
        return self._beta_delta

    alpha = property(fset=_set_alpha, fget=_get_alpha, fdel=None)
    beta = property(fset=_set_beta, fget=_get_beta, fdel=None)
    beta_delta = property(fset=_set_beta_delta, fget=_get_beta_delta, fdel=None)