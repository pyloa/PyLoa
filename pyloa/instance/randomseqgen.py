import numpy as np

from .seqgen import SequenceGenerator


class RandomSequenceGenerator(SequenceGenerator):
    """A SequenceGenerator that generates sequence_number many sequences of length sequence_size. Pages are picked
    equally distributed from the interval [min_page, max_page].

    Args:
        sequence_size (int): target size of each sequence
        sequence_number (int): number of sequences to generate till generator depletes
        min_page (int): minimum element within each generated sequence
        max_page (int): maximum element within each generated sequence
        dtype (type): type of each element within sequence
        seed (int): seed for random number generation

    Attributes:
        _rnd_state (np.random.RandomState): random number generator for self with own local seed
        _seed (int or None): seed _rnd_state was initialized with
    """
    # noinspection PyUnusedLocal
    def __init__(self,
                 sequence_size,
                 sequence_number,
                 min_page,
                 max_page,
                 dtype=np.int32,
                 seed=None,
                 **kwargs):
        super(RandomSequenceGenerator, self).__init__(
            sequence_size=sequence_size,
            sequence_number=sequence_number,
            min_page=min_page,
            max_page=max_page,
            dtype=dtype
        )
        self._rnd_state = np.random.RandomState(seed=seed)
        self._seed = seed
        self._logger.info("initialized {}".format(self))

    def _generate_instance(self):
        """Create and return a newly generated, random, equally distributed sequence

        Returns:
            np.ndarray: equally distributed sequence with shape=(_sequence_size,) and dtype=_dtype
        """
        return self._rnd_state.randint(
            low=self._min_page,
            high=self._max_page+1,
            size=self._sequence_size,
            dtype=self._dtype
        )

    def __str__(self):
        """str: string representation of self"""
        return self.__class__.__name__ + "(seed={})".format(self._seed)


class RandomDistributionSequenceGenerator(SequenceGenerator):
    """A SequenceGenerator that generates sequence_number many sequences of length sequence_size.

    A probability_distribution is initially rolled with probability_distribution = normal(sigma, mu) of shape
    (num_pages,). afterwards softmax is applied on probability_distribution. Pages are picked from the interval
    [min_page, max_page] according to the softmaxed probability_distribution.

    Args:
        sequence_size (int): target size of each sequence
        sequence_number (int): number of sequences to generate till generator depletes
        min_page (int): minimum element within each generated sequence
        max_page (int): maximum element within each generated sequence
        sigma (float): sigma for normal(sigma, mu) probability distribution
        mu (float): mu for normal(sigma, mu) probability distribution
        dtype (type): type of each element within sequence
        seed (int): seed for random number generation

    Attributes:
        _rnd_state (np.random.RandomState): random number generator for self with own local seed
        _seed (int or None): seed _rnd_state was initialized with
        _sigma (float): sigma for normal(sigma, mu) probability distribution
        _mu (float): mu for normal(sigma, mu) probability distribution
        _probability_distr (np.npdarray, shape=(_num_pages,), dtype=np.float32): probability distribution from softmaxed normal sample
    """
    # noinspection PyUnusedLocal
    def __init__(self,
                 sequence_size,
                 sequence_number,
                 min_page,
                 max_page,
                 sigma=1.0,
                 mu=0.0,
                 dtype=np.int32,
                 seed=None,
                 **kwargs):
        super(RandomDistributionSequenceGenerator, self).__init__(
            sequence_size=sequence_size,
            sequence_number=sequence_number,
            min_page=min_page,
            max_page=max_page,
            dtype=dtype
        )
        self._rnd_state = np.random.RandomState(seed=seed)
        self._seed = seed
        self._sigma = sigma
        self._mu = mu
        self._generate_probability_distr()
        self._logger.info("initialized {} with probability_distribution='{}'".format(self, self._probability_distr))

    def _generate_probability_distr(self):
        """rolls the distribution normal(sigma, mu) and applies a softmax to it"""
        # roll distribution
        self._probability_distr = self._sigma * self._rnd_state.randn(self._num_pages) + self._mu
        # apply softmax to distribution
        self._probability_distr = np.exp(self._probability_distr) / np.sum(np.exp(self._probability_distr))

    def _generate_instance(self):
        """Create and return a newly random, from normal(sigma, mu)-distributed softmaxed sample generated, sequence

        Returns:
            np.ndarray: sequence with shape=(_sequence_size,) and dtype=_dtype drawn from _probability_distr
        """
        return self._rnd_state.choice(a=self._pages, size=self._sequence_size, p=self._probability_distr)

    def _get_probability_distr(self):
        """np.ndarray: softmaxed probability distribution with shape=(_num_pages,) and dtype=np.float32"""
        return self._probability_distr

    def _set_probability_distr(self, value):
        assert isinstance(value, np.ndarray), "probability_distribution must be of type='{}'".format(np.ndarray)
        assert value.shape == (self._num_pages,), "probability_distribution must be of shape='{}' but " \
            "received shape='{}'".format((self._num_pages,), value.shape)
        assert np.sum(value) == 1.0, "probability_distribution's sum must equal 1.0"
        self._probability_distr = value

    def __str__(self):
        """str: string representation of self"""
        return self.__class__.__name__ + "(seed={}, \u03C3={}, \u03BC={})".format(self._seed, self._sigma, self._mu)

    probability_distribution = property(fset=_set_probability_distr, fget=_get_probability_distr, fdel=None)
