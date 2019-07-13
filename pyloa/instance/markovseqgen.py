import numpy as np

from .seqgen import SequenceGenerator


class MarkovChainSequenceGenerator(SequenceGenerator):
    """A SequenceGenerator that generates sequence_number many sequences of length sequence_size.

    A transition_matrix is initially rolled with transition_matrix = uniform(sigma, mu) of shape (num_pages, num_pages);
    whereas the j-th row of transition_matrix depicts the probability_dsitribution of page j. Afterwards softmax is
    applied on each row of transition_matrix. Pages are picked from the interval [min_page, max_page] according to the
    softmaxed probability_distribution of the last page picked. The initial last page to start generating a sequence is
    picked equally distributed.

    Args:
        sequence_size (int): target size of each sequence
        sequence_number (int): number of sequences to generate till generator depletes
        min_page (int): minimum element within each generated sequence
        max_page (int): maximum element within each generated sequence
        sigma (float): sigma for uniform(sigma, mu) probability distribution
        mu (float): mu for uniform(sigma, mu) probability distribution
        dtype (type): type of each element within sequence,
        seed (int): seed for random number generation

    Attributes:
        _rnd_state (np.random.RandomState): random number generator for self with own local seed
        _seed (int or None): seed _rnd_state was initialized with
        _sigma (float): sigma for uniform(sigma, mu) probability distribution
        _mu (float): mu for uniform(sigma, mu) probability distribution
        _transition_matrix (np.npdarray, shape=(_num_pages,_num_pages), dtype=np.float32): a matrix where the j-th row
            depicts j-th page's softmaxed uniform(sigma, mu) distribution to transition with to following pages
    """
    # noinspection PyUnusedLocal
    def __init__(self,
                 sequence_size,
                 sequence_number,
                 min_page,
                 max_page,
                 sigma=1,
                 mu=0,
                 dtype=np.int32,
                 seed=None,
                 **kwargs):
        super(MarkovChainSequenceGenerator, self).__init__(
            sequence_size=sequence_size,
            sequence_number=sequence_number,
            min_page=min_page,
            max_page=max_page,
        )
        self._rnd_state = np.random.RandomState(seed=seed)
        self._seed = seed
        self._sigma = sigma
        self._mu = mu
        self._generate_matrix()
        self._logger.info("initialized {} with transition_matrix='{}'".format(self, self._transition_matrix))

    def _generate_matrix(self):
        """rolls the _transition_matrix with uniform(sigma, mu) and applies a softmax to each row"""
        # roll transition matrix
        self._transition_matrix = self._sigma * self._rnd_state.randn(self._num_pages, self._num_pages) + self._mu
        # apply softmax normalization to each row
        for idx, val in enumerate(self._transition_matrix):
            _exp = np.exp(val)
            self._transition_matrix[idx, :] = _exp / np.sum(_exp)

    def _generate_instance(self):
        """Create and return a newly generated sequence.

        This function initially picks a last_page equally distributed; based on last_page's transition (row entry) the
        following page in sequence is selected, which once again will become last_page. This process is repeated untill
        the sequence is of size sequence_size.

        Returns:
            np.ndarray: sequence with shape=(_sequence_size,) and dtype=_dtype generated from _transition_matrix
        """
        res = np.empty(self._sequence_size, dtype=self._dtype)
        # randomly roll equally distributed what the initial page is
        _last_page = self._rnd_state.randint(low=self._min_page, high=self._max_page+1)
        for idx in range(self._sequence_size):
            # pick next page from _last_page's transitions
            res[idx] = self._rnd_state.choice(self._pages, size=1, p=self._transition_matrix[_last_page-1])
            _last_page = res[idx]
        return res

    def _get_transition_matrix(self):
        """np.ndarray: matrix to draw sequences from with shape=(_num_pages,_num_pages) and dtype=np.float32"""
        return self._transition_matrix

    def _set_transition_matrix(self, value):
        assert isinstance(value, np.ndarray), "transition_matrix must be of type='{}'".format(np.ndarray)
        assert value.shape == (self._num_pages, self._num_pages), "transition_matrix must be of shape='{}' but " \
            "received shape='{}'".format((self._num_pages, self._num_pages), value.shape)
        for idx, row in enumerate(value):
            assert np.sum(row) == 1.0, "summed row='{}' at index='{}' must equal 1.0".format(row, idx)
        self._transition_matrix = value

    def __str__(self):
        """str: string representation of self"""
        return self.__class__.__name__ + "(seed={}, \u03C3={}, \u03BC={})".format(self._seed, self._sigma, self._mu)

    transition_matrix = property(fset=_set_transition_matrix, fget=_get_transition_matrix, fdel=None)

