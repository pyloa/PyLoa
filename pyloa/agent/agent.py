import logging
from abc import abstractmethod, ABCMeta


class Agent(metaclass=ABCMeta):
    """Abstract base class for Agents.

    Args:
        state_dim (int): defines input shape=(_state_dim,) of input states
        action_dim (int): defines the number of different actions the agent can take
        name (str): depicts a name for the agent

    Attributes:
        _logger (logging.Logger): logger instance for each subclassed instance
        _state_dim (int): defines input shape=(_state_dim,) of input states
        _action_dim (int): defines the number of different actions the agent can take
        _name (str): depicts a name for the agent
    """
    def __init__(self, state_dim, action_dim, name=None):
        self._name = self.__class__.__name__ if name is None else name
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._logger = logging.getLogger(__name__)
        self._logger.info("initializing {}".format(self))

    @abstractmethod
    def act(self, state):
        """Agent decides on an action based on information given with state.

        Args:
            state (np.ndarray with shape=(_state_dim,) or dict): input state

        Returns:
            int: action to be taken

        Raises:
            NotImplementedError: if abstract method is to be invoked
        """
        raise NotImplementedError("Must be implemented in subclass.")

    @abstractmethod
    def play_instance(self, environment, episode, is_train, writer, invert_cost=True):
        """Agent plays an instance on environment untill environment terminates.

        Args:
            environment (pyloa.environment.Environment): environment to play on
            episode (int): number depicting how many episodes agent played so far
            is_train (bool): depicts if agent should train (True) or just evaluate (False)
            writer (tf.summary.FileWriter): tensorboard logging for playing a instance
            invert_cost (bool): depicts if costs are inverted after calculation (cost = negative reward)

        Returns:
            int: action to be taken

        Raises:
            NotImplementedError: if abstract method is to be invoked
        """
        raise NotImplementedError("Must be implemented in subclass.")

    def __str__(self):
        """str: string representation of agent"""
        return self._name
