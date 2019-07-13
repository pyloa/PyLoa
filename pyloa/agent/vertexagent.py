from abc import abstractmethod

import numpy as np

from .agent import Agent
from .rlagent import RLAgent
from .qagent import QAgent
from pyloa.utils.tensorboard import tensorboard_array, tensorboard_scalar, tensorboard_histo


class VertexAgent(Agent):
    """Abstract base class for Agents that play on pyloa.environment.VertexEnvironment instances.

    Args:
        state_dim (int): defines input shape=(_state_dim,) of input states
        action_dim (int): defines the number of different actions the agent can take
    """
    def __init__(self, state_dim, action_dim, **kwargs):
        super().__init__(state_dim=state_dim, action_dim=action_dim)

    # noinspection PyUnresolvedReferences
    # noinspection PyAttributeOutsideInit
    def play_instance(self, environment, is_train, episode, writer, invert_cost=True):
        """Agent plays on environment for one episode (on one graph).

        Args:
            environment (pyloa.environment.VertexEnvironment): environment to play on
            episode (int): number depicting how many episodes agent played so far
            is_train (bool): depicts if agent should train (True) or just evaluate (False)
            writer (tf.summary.FileWriter): tensorboard logging for playing an instance
            invert_cost (bool): depicts if costs are inverted after calculation (cost = negative reward)

        Returns:
            tuple: agent will return its calculated metrics of this episode as follows (steps, cost, page_faults)
                    steps (int): depicts the number of actions till the environment terminated
                    cost (int): total cost accumulated during episode
                    page_faults (int): depicts the number of page faults the agent had during episode

        Raises:
            RuntimeError: if agent is invoked with is_train=True, but agent is not a subclass of RLAgent
        """
        _is_trainable = isinstance(self, RLAgent)         # has epsilon and train
        _has_memory = isinstance(self, QAgent)            # has epsilon, memory and train
        # temp vars
        _temp_cost = 0
        _temp_actions = []
        _temp_losses = []
        _temp_epsilon = 0.0
        _temp_local_cognition_fault = 0

        # sanity check to make sure only trainable agents can be trained
        if is_train and not _is_trainable:
            raise RuntimeError("{} called to learn but agent isn't trainable!".format(self.__class__.__name__))

        # change epsilon if agent is trainable but isn't training right now for eval
        if _is_trainable and not is_train:
            _temp_epsilon = self.epsilon
            self.epsilon = 1.0

        # play environment till it terminates
        while not environment.is_terminated:
            # get current state
            state = environment.get_state(vectorized_state=_is_trainable)

            # get next predicted action
            action = self.act(state=state)

            # take action
            next_state, cost = environment.step(action=action, vectorized_state=_is_trainable)
            cost = -cost if invert_cost else cost

            # update action_histogram, cost temp metrics, time in cache
            _temp_local_cognition_fault += (not environment.was_locally_colorized)
            _temp_actions.append(action)
            _temp_cost += abs(cost)

            # store action if agent is training
            if _has_memory and is_train:
                self.store(state=state, action=action, reward=cost, next_state=next_state)

            # train episodic non-QAgents, append loss values
            if not _has_memory and _is_trainable and is_train:
                _loss = self.train(state=state, action=action, reward=cost, next_state=next_state)
                _temp_losses.append(_loss)

        # train replay memory QAgents
        if _has_memory and is_train:
            _temp_losses = self.train(train_writer=writer)

        # adapt epsilon
        if _is_trainable:
            if is_train:
                self.increment_epsilon()        # increment epsilon if agent is trainable and is training
            else:
                self.epsilon = _temp_epsilon    # change epsilon back if agent is trainable but isn't training right now

        # store metrics
        if writer is not None:
            tensorboard_scalar(writer, 'total_cost/' + str(self), _temp_cost, episode)
            tensorboard_scalar(writer, 'avg_cost/' + str(self), _temp_cost / environment.num_steps, episode)
            tensorboard_scalar(writer, 'local_cognition_fault_ratio/' + str(self), _temp_local_cognition_fault /
                               environment.num_steps, episode)
            tensorboard_scalar(writer, 'num_colors/' + str(self), environment.num_colors, episode)
            tensorboard_scalar(writer, 'global_colorizing/' + str(self), environment.is_colorized, episode)
            tensorboard_histo(writer, 'actions/' + str(self), _temp_actions, episode)
            if is_train:
                tensorboard_array(writer, 'episode/loss', _temp_losses, episode)
                tensorboard_scalar(writer, 'episode/epsilon', self.epsilon, episode)

        # return performance
        return environment.num_steps, _temp_cost, 0

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


class FIFOVertexAgent(VertexAgent):
    """Deterministic First-In-First-Out VertexAgent.

    Args:
        state_dim (int): defines input shape=(_state_dim,) of input states
        action_dim (int): defines the number of different actions the agent can take
    """
    def __init__(self, state_dim, action_dim, **kwargs):
        super().__init__(state_dim=state_dim, action_dim=action_dim, **kwargs)

    def act(self, state):
        """Agent uses smallest-numerical color, which is unused in neighbourhood of order 1.

        Args:
            state (dict): input state

        Returns:
            int: action to be taken

        Raises:
            AssertionError: if current graph is not colorable by FIFOVertexAgent
        """
        _all_colors = list(range(1, 1+len(state['color_flag'])))        # create all colors
        _neighbour_colors = set(state['delta'][state['delta'] > 0])     # get neighbourhood colors
        for n in _neighbour_colors:                                     # remove neighbourhood colors
            _all_colors.remove(n)
        assert len(_all_colors) > 0, "{} can not color current graph"
        return _all_colors[0] - 1                                       # return lowest color


class RandomVertexAgent(VertexAgent):
    """Non-Deterministic RandomVertexAgent, which picks a random color each time.

    Args:
        state_dim (int): defines input shape=(_state_dim,) of input states
        action_dim (int): defines the number of different actions the agent can take
    """
    def __init__(self, state_dim, action_dim, seed=None, **kwargs):
        self._seed = seed
        super().__init__(state_dim=state_dim, action_dim=action_dim, **kwargs)
        self._rnd_state = np.random.RandomState(seed=seed)

    def act(self, state):
        """Agent picks a random available color.

        Args:
            state (dict): input state

        Returns:
            int: action to be taken
        """
        return self._rnd_state.randint(low=0, high=len(state['color_flag']))

    def __str__(self):
        """str: string representation of self"""
        return self.__class__.__name__ + "(seed={})".format(self._seed)