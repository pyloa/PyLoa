import os
import pickle as pk

import numpy as np

from .rlagent import RLAgent
from .pagingagent import PagingAgent


class QTableAgent(RLAgent, PagingAgent):
    """A simple Q-Learning table agent with dict as a table container. This is a naive implementation for demonstration
    purposes only. Use QTableAgent only on toy examples, elsewise you will run out of memory.

    Args:
        state_dim (int): defines input shape=(_state_dim,) of input states
        action_dim (int): defines the number of different actions the agent can take
        learning_rate (float): learning_rate of agent's model
        discount_factor (float): value at which future Q-target Values are discounted
        epsilon (float): agent's current chance to exploit his knowledge, else explore action_space
        epsilon_delta (float): increment value of epsilon for each increment call
        epsilon_max (float): maximum value of epsilon
        save_file (str): location at which agent's checkpoints are stored at

    Attributes:
        _rnd_state (np.random.RandomState): random number generator for self with own local seed
        _discount_factor (float): value at which Q-target Values are discounted
        _q_table (dict): key type=str as state, value type=np.ndarray shape=(_action_dim,) as Q-Values for each action
    """
    # noinspection PyUnusedLocal
    def __init__(self,
                 state_dim,
                 action_dim,
                 learning_rate,
                 discount_factor,
                 epsilon=0.90,
                 epsilon_delta=50e-4,
                 epsilon_max=1.0,
                 save_file=None,
                 seed=None,
                 **kwargs):
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=learning_rate,
            epsilon=epsilon,
            epsilon_delta=epsilon_delta,
            epsilon_max=epsilon_max,
            save_file=save_file
        )
        # params
        self._rnd_state = np.random.RandomState(seed=seed)
        self._discount_factor = discount_factor
        self._q_table = None
        # init
        self._build_model()

    def _build_model(self):
        """initialize a dictionary used as a lookup table as model."""
        self._q_table = dict()

    def act(self, state):
        """QTableAgent decides on an action based on information given with state.

        Args:
            state (np.ndarray, shape=(_state_dim,), dtype=np.int32): input state

        Returns:
            int: action to be taken
        """
        _h_state = str(state)                                                           # np.ndarray -> str
        if _h_state not in self._q_table:                                               # state not seen yet
            self._q_table[_h_state] = np.zeros(self._action_dim, dtype=np.int32)        # initial Q-Values are 0
            action = self._rnd_state.randint(0, self._action_dim)                       # explore random action
        elif self._rnd_state.uniform(low=0.0, high=1.0) < self._epsilon:                # e-greedy exploitation
            action = np.argmax(self._q_table[_h_state])                                 # pick best action
        else:                                                                           # (1-e) exploration
            action = self._rnd_state.randint(0, self._action_dim)                       # explore random action
        return action

    # noinspection PyMethodOverriding
    def train(self, state, action, reward, next_state):
        """QTableAgent trains.

        Args:
            state (np.ndarray, shape=(_state_dim,), dtype=np.int32): observed initial state before action
            action (int): action selected for state
            reward (float): reward from environment for taking action in state
            next_state: (np.ndarray, shape=(_state_dim,), dtype=np.int32): observed state after action

        Returns:
            int: number of states explored by QTableAgent
        """
        _h_state, _h_nstate = str(state), str(next_state)                               # np.ndarray -> str
        if _h_nstate not in self._q_table:                                              # next state not seen yet
            self._q_table[_h_nstate] = np.zeros(self._action_dim, dtype=np.int32)       # initial Q-Values are 0
        # update Q-Values for action
        self._q_table[_h_state][action] = self._q_table[_h_state][action] + self._learning_rate * \
            (reward + self._discount_factor * np.max(self._q_table[_h_nstate]) - self._q_table[_h_state][action])
        return len(self._q_table)

    def save(self, file_path=None):
        """Save agent's Q-Table on persistent storage.

        Args:
            file_path (str): existing path to file
        """
        file_path = file_path if file_path is not None else self._save_file
        with open(file_path + '.pkl', 'wb') as f:
            pk.dump(self._q_table, f, pk.HIGHEST_PROTOCOL)
        self._logger.info("Saved model {} in {}".format(self, file_path))

    def store(self, state, action, reward, next_state):
        """QTableAgent does not have a memory to store in."""
        pass

    def load(self, file_path=None):
        """Load agent's Q-Table from persistent storage.

        Args:
            file_path (str): existing path to picked Q-Table
        """
        file_path = file_path if file_path is not None else self._save_file
        directory = os.path.dirname(file_path) if not os.path.isdir(file_path) else file_path
        try:
            with open(os.path.join(directory, self.__class__.__name__ + '.pkl'), 'rb') as f:
                self._q_table = pk.load(f)
            self._logger.info("Loaded model {} from {}".format(self, file_path))
        except Exception as e:
            self._logger.error("Couldn't load model {} from file='{}': {}".format(self, file_path, str(e)))
            self._logger.info("model {} starts with default initialization".format(self))
