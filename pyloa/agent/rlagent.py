import os
from abc import abstractmethod

import tensorflow as tf
import numpy as np

from .agent import Agent


class RLAgent(Agent):
    """Abstract base class for Agents that play and learn on pyloa.environment.PagingEnvironment instances.

    Args:
        state_dim (int): defines input shape=(_state_dim,) of input states
        action_dim (int): defines the number of different actions the agent can take
        learning_rate (float): learning_rate of agent's model
        epsilon (float): agent's current chance to exploit his knowledge, else explore action_space
        epsilon_delta (float): increment value of epsilon for each increment call
        epsilon_max (float): maximum value of epsilon
        save_file (str): location at which agent's checkpoints are stored at

    Attributes:
        _learning_rate (float): learning_rate of agent or agent's model
        _epsilon (float): agent's current chance to exploit his knowledge, else explore action_space
        _epsilon_delta (float): increment value of epsilon for each increment call
        _epsilon_max (float): maximum value of epsilon
        _save_file (str): location at which agent's checkpoints are stored at
        _sess (tf.python.client.session.Session): tensorflow session handle
        _global_step (tf.python.ops.variables.RefVariable): global train steps for agent's model
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 learning_rate,
                 epsilon,
                 epsilon_delta,
                 epsilon_max=1.0,
                 save_file=None,
                 **kwargs):
        super().__init__(state_dim=state_dim, action_dim=action_dim, **kwargs)
        # params
        self._learning_rate = learning_rate
        self._epsilon = epsilon
        self._epsilon_delta = epsilon_delta
        self._epsilon_max = epsilon_max
        self._save_file = save_file
        # tensorflow
        self._sess = tf.Session()
        self._global_step = tf.Variable(0, name='global_step', trainable=False)

    @abstractmethod
    def _build_model(self):
        """Build the model: e.g. a tensorflow computational graph.

        Raises:
            NotImplementedError: if abstract method is to be invoked
        """
        raise NotImplementedError("Must be implemented in subclass.")

    @abstractmethod
    def act(self, state):
        """Agent decides on an action based on information given with state.

        Args:
            state (np.ndarray with shape=(_state_dim,)): input state

        Returns:
            int: action to be taken

        Raises:
            NotImplementedError: if abstract method is to be invoked
        """
        raise NotImplementedError("Must be implemented in subclass.")

    @abstractmethod
    def train(self):
        """Agent trains model.

        Raises:
            NotImplementedError: if abstract method is to be invoked

        Returns:
            float: loss of train step
        """
        raise NotImplementedError("Must be implemented in subclass.")

    @abstractmethod
    def store(self, state, action, reward, next_state):
        """Store quadruple of experience.

        Args:
            state (np.ndarray with shape=(_state_dim,): observed initial state before action
            action (int): action selected for state
            reward (float): reward from environment for taking action in state
            next_state (np.ndarray with shape=(_state_dim,)): observed state after action
        """
        raise NotImplementedError("Must be implemented in subclass.")

    def save(self, file_path=None):
        """Save agent's model with _global_step as suffix on persistent storage.

        Args:
            file_path (str): existing path to file
        """
        file_path = file_path if file_path is not None else self._save_file
        saver = tf.train.Saver()
        path = saver.save(self._sess, file_path, global_step=self._global_step)
        self._logger.info("Saved model {} in {}".format(self.__class__.__name__, path))

    def load(self, file_path=None):
        """Load agent's last checkpoint from persistent storage.

        Args:
            file_path (str): existing path to checkpoint
        """
        file_path = file_path if file_path is not None else self._save_file
        _cp_file = tf.train.latest_checkpoint(os.path.dirname(file_path))
        saver = tf.train.Saver()
        try:
            saver.restore(self._sess, _cp_file)
            self._logger.info("Loaded model {} from {}".format(self, _cp_file))
        except Exception as e:
            self._logger.error("Couldn't load model {} from file='{}': {}".format(self, _cp_file, str(e)))
            self._logger.info("model {} starts with default initialization".format(self))

    def increment_epsilon(self):
        """Increase agent's epsilon by _epsilon_delta without exceeding _epsilon_max."""
        self.epsilon = min(self._epsilon + self._epsilon_delta, self._epsilon_max)

    def _get_epsilon(self):
        """float: current chance of agent to exploit his current knowledge, else explore action_space"""
        return self._epsilon

    def _set_epsilon(self, value):
        assert isinstance(value, (float, np.float)), "epsilon must be of type '{}' or '{}' but found type='{}' instead"\
            .format(float, np.float, type(value))
        assert 0.0 <= value <= 1.0, "epsilon must be in interval [0.0, 1.0] but found value='{}' instead".format(value)
        self._epsilon = value

    epsilon = property(fset=_set_epsilon, fget=_get_epsilon, fdel=None)
