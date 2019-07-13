from abc import abstractmethod

import numpy as np

from .agent import Agent
from .qagent import QAgent
from .rlagent import RLAgent
from pyloa.utils.tensorboard import tensorboard_scalar, tensorboard_array, tensorboard_histo


class PagingAgent(Agent):
    """Abstract base class for Agents that play on pyloa.environment.PagingEnvironment sequences.

    Args:
        state_dim (int): defines input shape=(_state_dim,) of input states
        action_dim (int): defines the number of different actions the agent can take
        name (str): depicts a name for the agent
    """
    def __init__(self, state_dim, action_dim, name=None, **kwargs):
        super().__init__(state_dim=state_dim, action_dim=action_dim, name=name)

    # noinspection PyUnresolvedReferences
    # noinspection PyAttributeOutsideInit
    def play_instance(self, environment, is_train, episode, writer, invert_cost=True):
        """Agent plays on environment for one episode (on one sequence).

        Args:
            environment (pyloa.environment.PagingEnvironment): environment to play on
            episode (int): number depicting how many episodes agent played so far
            is_train (bool): depicts if agent should train (True) or just evaluate (False)
            writer (tf.summary.FileWriter): tensorboard logging for playing a sequence
            invert_cost (bool): depicts if costs are inverted after calculation (cost = negative reward)

        Returns:
            tuple: agent will return its calculated metrics of this episode as follows (steps, cost, page_faults)
                    steps (int): depicts the number of actions till the environment terminated
                    cost (int): total cost accumulated during episode
                    page_faults (int): depicts the number of page faults the agent had during episode

        Raises:
            RuntimeError: if agent is invoked with is_train=True, but agent is not a subclass of RLAgent
        """
        _is_trainable = isinstance(self, RLAgent)               # has epsilon and train
        _has_memory = isinstance(self, QAgent)                  # has epsilon, memory and train
        # temp vars
        _temp_cost = 0
        _temp_num_page_fault = 0
        _temp_actions = []
        _temp_losses = []
        _temp_epsilon = 0.0
        _temp_num_cognition_fault = 0

        # sanity check to make sure only trainable agents can be trained
        if is_train and not _is_trainable:
            raise RuntimeError("{} called to learn but agent isn't trainable!".format(self.__class__.__name__))

        # change epsilon if agent is trainable but isn't training right now for eval
        if _is_trainable and not is_train:
            _temp_epsilon = self.epsilon
            self.epsilon = 1.0

        # play environment till it terminates
        while not environment.is_terminated:
            # get current state (trainable agents expect vecotrized np.ndarray states)
            state = environment.get_state(vectorized_state=_is_trainable)

            # get next predicted action
            action = self.act(state=state)

            # update page fault metrics: a page fault occurred
            _temp_num_page_fault += environment.is_pagefault
            # update cognition fault metrics: agent took action even though he shouldn't have OR
            # agent took no action even though he should have)
            _temp_num_cognition_fault += environment.is_pagefault ^ (action > 0)

            # take action, invert cost
            next_state, cost = environment.step(action=action, vectorized_state=_is_trainable)
            cost = -cost if invert_cost else cost

            # update action_histogram, cost temp metrics
            _temp_actions.append(action)
            _temp_cost += abs(cost)

            # store action if agent is training and has memory
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
            tensorboard_scalar(writer, 'miss_ratio/' + str(self), _temp_num_page_fault / environment.num_steps, episode)
            tensorboard_scalar(writer, 'cognition_fault_ratio/' + str(self), _temp_num_cognition_fault /
                               environment.num_steps, episode)
            tensorboard_scalar(writer, 'error_ratio/' + str(self), (_temp_num_page_fault + _temp_num_cognition_fault) /
                               environment.num_steps, episode)
            tensorboard_histo(writer, 'actions/' + str(self), _temp_actions, episode)
            if is_train:
                tensorboard_array(writer, 'episode/loss', _temp_losses, episode)
                tensorboard_scalar(writer, 'episode/epsilon', self.epsilon, episode)

        # return performance
        return environment.num_steps, _temp_cost, _temp_num_page_fault

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


class FIFOAgent(PagingAgent):
    """Deterministic First-In-First-Out agent; on-line, conservative, theoretical competitive ratio = cache_size"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def act(self, state):
        """Agent evicts oldest page in cache.

        Args:
            state (dict): input state

        Returns:
            int: action to be taken
        """
        if not state['is_pagefault']:                   # if the current state is not a page fault
            return 0                                    # don't evict a slot (0 = no action)
        return np.argmax(state['time_in_cache']) + 1    # evict oldest page with action 'index+1'


class LIFOAgent(PagingAgent):
    """Deterministic Last-In-First-Out agent; on-line, non-conservative, theoretical competitive ratio is unbound"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def act(self, state):
        """Agent evicts youngest page in cache,

        Args:
            state (dict): input state

        Returns:
            int: action to be taken
        """
        if not state['is_pagefault']:                   # if the current state is not a page fault
            return 0                                    # don't evict a slot (0 = no action)
        return np.argmin(state['time_in_cache']) + 1    # evict youngest page with action 'index+1'


class MostFrequentlyUsedAgent(PagingAgent):
    """Deterministic Most-Frequently-Used agent; on-line, non-conservative, theo. competitive ratio = cache_size"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def act(self, state):
        """Agent evicts most-accessed page in cache,

        Args:
            state (dict): input state

        Returns:
            int: action to be taken
        """
        if not state['is_pagefault']:                   # if the current state is not a page fault
            return 0                                    # don't evict a slot (0 = no action)
        return np.argmax(state['histogram']) + 1        # evict most-accessed page with action 'index+1'


class LeastFrequentlyUsedAgent(PagingAgent):
    """Deterministic Least-Frequently-Used agent; on-line, non-conservative, theoretical competitive ratio is unbound"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def act(self, state):
        """Agent evicts least-accessed page in cache.

        Args:
            state (dict): input state

        Returns:
            int: action to be taken
        """
        if not state['is_pagefault']:                   # if the current state is not a page fault
            return 0                                    # don't evict a slot (0 = no action)
        return np.argmin(state['histogram']) + 1        # evict least-accessed oage with action 'index+1'


class MostRecentlyUsedAgent(PagingAgent):
    """Deterministic Most-Recently-Used agent; on-line, non-conservative, theoretical competitive ratio is unbound"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def act(self, state):
        """Agent evicts most-recently-accessed page in cache.

        Args:
            state (dict): input state

        Returns:
            int: action to be taken
        """
        if not state['is_pagefault']:                       # if the current state is not a page fault
            return 0                                        # don't evict a slot (0 = no action)
        return np.argmin(state['time_since_touched']) + 1   # evict most-recently-accessed page with action 'index+1'


class LeastRecentlyUsedAgent(PagingAgent):
    """Deterministic Least-Recently-Used agent; on-line, non-conservative, theoretical competitive ratio = cache_size"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def act(self, state):
        """Agent evicts least-recently-accessed page in cache.

        Args:
            state (dict): input state

        Returns:
            int: action to be taken
        """
        if not state['is_pagefault']:                       # if the current state is not a page fault
            return 0                                        # don't evict a slot (0 = no action)
        return np.argmax(state['time_since_touched']) + 1   # evict least-recently-accessed page with action 'index+1'


class RandomAgent(PagingAgent):
    """Non-Deterministic randomized agent; on-line, non-conservative, theoretical competitive ratio is unclear

    Args:
        seed (int): sets np.random.seed for reproducible randomization

    Attributes:
        _rnd_state (np.random.RandomState): random number generator for self with own local seed
        _seed (int):  sets np.random.seed for reproducible randomization
    """
    # noinspection PyUnusedLocal
    def __init__(self, seed=None, **kwargs):
        self._seed = seed
        self._rnd_state = np.random.RandomState(seed=seed)
        super().__init__(**kwargs)

    def act(self, state):
        """Agent evicts least-recently-accessed page in cache,

        Args:
            state (dict): input state

        Returns:
            int: action to be taken
        """
        if not state['is_pagefault']:                                   # if the current state is not a page fault
            return 0                                                    # don't evict a slot (0 = no action)
        return self._rnd_state.randint(low=0, high=len(state['cache'])) + 1   # evict randomly chosen 'index+1'

    def __str__(self):
        """str: string representation of self"""
        return self.__class__.__name__ + "(seed={})".format(self._seed)


class BeladysOptimalAgent(PagingAgent):
    """Deterministic optimal agent; off-line, conservative, theoretical competitive ratio = 1.0

    Belady's Algorithm is the optimal, off-line algorithm for the paging problem. The agent overloads super's
    play_instance to calculate an off-line solution via _calc_opt - inspecting the sequence that environment is going to
    iterate over (looking into the future). Within _calc_opt the agent fills its list actions with the optimal action to
    take at each time step. Afterwards, the agent invokes super's play_instance, whereas each function call of act
    utilizes the off-line information gathered from _calc_opt.

    Args:
        state_dim (int): defines input shape=(_state_dim,) of input states
        action_dim (int): defines the number of different actions the agent can take

    Attributes:
        _current_time_step (int): time step that increases after each act and resets for a new play_instance call
        _actions (list of int): optimal action to take at each time step
    """
    # noinspection PyUnusedLocal
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._current_time_step = 0
        self._actions = list()

    def _calc_opt(self, sequence, cache):
        """Calculates optimal action at each time step to play sequence, such that number of page faults is minimized.

        Args:
            sequence (np.ndarray): the sequence that the pyloa.environment.PagingEnvironment instance is going to play
            cache (np.ndarray): the initial cache of the pyloa.environment.PagingEnvironment instance
        """
        cache = np.copy(cache)                                           # cache-copy for simulating self._actions
        time_distance = np.zeros(len(cache), dtype=np.int32)             # temp placeholder for future time_steps
        for time_step, request in enumerate(sequence):                   # for request at time_step
            if request not in cache:                                     # if request not yet in cache
                for index, page in enumerate(cache):                     # for page in cache at index
                    if page == 0 or page not in sequence[time_step+1:]:  # page is init-value or not seen in future
                        self._actions.append(index)                      # -> index is best to evict (flush index)
                        break                                            # magic for-break-else construct (check ref)
                else:                                                    # else case evaluated iff 'break' didn't happen
                    for index, page in enumerate(cache):                 # for page in cache at index
                        # get all future time_steps at which page occurs in future
                        future_time_steps = np.argwhere(sequence[time_step+1:] == page)[0]
                        # if occurence exists, set to first seen time_step, else 'inf'
                        first_occurence = future_time_steps[0] if len(future_time_steps) > 0 else np.iinfo(np.int32).max
                        time_distance[index] = first_occurence
                    # flush index furthest away in the future
                    self._actions.append(np.argmax(time_distance))
                # simulate evict at index
                cache[self._actions[time_step]] = request
            else:                                                        # request is in cache -> do nothing
                self._actions.append(-1)

    # noinspection PyMethodOverriding
    def play_instance(self, environment, episode, writer, invert_cost=True, **kwargs):
        """Agent plays on environment for one episode (on one sequence) by firstly calculating an off-line solution and
        secondly invoking super's play_instance to play it sequentially while 'cheating' with off-line information in
        each function call of act.

        Args:
            environment (pyloa.environment.PagingEnvironment): environment to play on
            episode (int): number depicting how many episodes agent played so far
            writer (tf.summary.FileWriter): tensorboard logging for play sequence
            invert_cost (bool): depicts if costs are inverted after calculation (cost = negative reward)

        Returns:
            tuple: agent will return its calculated metrics of this episode as follows (steps, cost, page_faults)
                    steps (int): depicts the number of actions till the environment terminated
                    cost (int): total cost accumulated during episode
                    page_faults (int): depicts the number of page faults the agent had during episode
        """
        # reset counter, clear last optimal solution
        self._current_time_step = 0
        self._actions.clear()
        # get sequence from environment, calc optimal offline solution on sequence such that self.act is `cheating`
        self._calc_opt(environment.instance, environment.cache)
        # call super-class method on self to emulate steps while tracking metrics
        return PagingAgent.play_instance(self, environment=environment, episode=episode, writer=writer, is_train=False,
                                         invert_cost=invert_cost)

    def act(self, **kwargs):
        """Agent evicts optimal page in cache.

        Returns:
            int: action to be taken
        """
        action = self._actions[self._current_time_step] + 1         # evict page with action 'index+1'
        self._current_time_step += 1                                # increase time step
        return action
