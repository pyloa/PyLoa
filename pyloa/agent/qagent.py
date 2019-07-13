from abc import abstractmethod

import tensorflow as tf
import numpy as np

from .rlagent import RLAgent
from pyloa.utils.memory import Memory, ReplayMemory, PrioritizedReplayMemory


class QAgent(RLAgent):
    """Abstract base class for Agents that play and learn on pyloa.environment.PagingEnvironment with Q-Learning on a
    replay memory according to Mnih et al. in https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf

    QAgent defines an abstract class for Deep-Q-Networks (DQN); implementations of QAgent have to implement the function
    _build_q_network, which has to return the output of the network (Q_Values). QAgent invokes _build_q_network twice
    while building the model; firstly, to create the Q-target Network and secondly to create the Q-eval Network - both
    have the same structure according to the child's implementation. Q-target's variables are frozen and won't be
    updated during _train_op; only Q-eval will be trained. Every _replace_threshold-many train steps the operation
    _replace_params is invoked, which replaces Q-target's variables with Q-eval's variables. Each entry within the
    agent's replay pyloa.utils.Memory is a quadruple of (state, action, reward, next_state); each train step draws
    uniformly a random a sample of size _sample_size from replay memory. Each sample is batched with batch_size. The
    _loss function for training is defined as:

        _loss = (reward + _discount_factor * max(q_target(next_state)) - q_eval(state)) ** 2

    whereas q_eval's entry at index=action (from memory) is used and q_target's entry at index=argmax is used. The _loss
    function is L2-regularized with _l2_scale for all variables in Q-eval with:

        for var in q_eval_variables:
             _loss += _l2_scale * tf.nn.l2_loss(var)

    Args:
        state_dim (int): defines input shape=(_state_dim,) of input states
        action_dim (int): defines the number of different actions the agent can take
        learning_rate (float): learning_rate of agent's model
        epsilon (float): agent's current chance to exploit his knowledge, else explore action_space
        epsilon_delta (float): increment value of epsilon for each increment call
        epsilon_max (float): maximum value of epsilon
        discount_factor (float): value at which future Q-target Values are discounted
        memory_size (int): total size of agent's replay memory
        sample_size (int): size of sample randomly drawn from memory to train on per train step
        batch_size (int): size of each batch a sample for training is batched with
        l2_scale (float): L2-Regularization scale for eval network
        replace_threshold (int): depicts after how many train steps Q-target's variables are replaced with Q-eval's
        save_file (str): location at which agent's checkpoints are stored at

    Attributes:
        _discount_factor (float): value at which Q-target Values are discounted
        _memory_size (int): total size of agent's replay memory
        _sample_size (int): size of sample randomly drawn from memory to train on per train step
        _batch_size (int): size of each batch a sample for training is batched with
        _l2_scale (float): L2-Regularization scale for eval network
        _rnd_state (np.random.RandomState): random number generator for self with own local seed
        _replace_step (int): counting variable from 0 to _replace_threshold
        _memory (pyloa.utils.Memory): replay memory to store in and draw from
        _state (tf.Tensor): placeholder with shape=(?, _state_dim) for _batch_size many states
        _next_state (tf.Tensor): placeholder shape=(?, _state_dim) for _batch_size many next_states
        _reward (tf.Tensor): placeholder shape=(?,) for _batch_size many rewards
        _action (tf.Tensor): placeholder shape=(?,) for _batch_size many actions
        _loss (tf.Tensor): as defined above in __class__.__doc__
        _train_op (tf.Operation): minimizes _loss with RMSPropOptimizer with respect to _learning_rate
        _eval_outs: (tf.Tensor): Q-eval Values Tensor for inference on model (invoked from act)
        _eval_params (list of tf.Variable): Q-eval Network variables
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 learning_rate,
                 epsilon,
                 epsilon_delta,
                 epsilon_max,
                 discount_factor,
                 memory_size,
                 sample_size,
                 batch_size,
                 l2_scale,
                 replace_threshold,
                 seed=None,
                 save_file=None,
                 **kwargs):
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=learning_rate,
            epsilon=epsilon,
            epsilon_delta=epsilon_delta,
            epsilon_max=epsilon_max,
            save_file=save_file,
            **kwargs
        )
        # params
        self._discount_factor = discount_factor
        self._memory_size = memory_size
        self._sample_size = sample_size
        self._batch_size = batch_size
        self._l2_scale = l2_scale
        self._replace_threshold = replace_threshold
        # vars
        self._rnd_state = np.random.RandomState(seed=seed)
        self._replace_step = 0
        if 'memory' in kwargs and 'type' in kwargs['memory']:
            if not issubclass(kwargs['memory']['type'], Memory):
                raise RuntimeError("optional memory implementation must subclass '{}'".format(Memory))
            self._memory = kwargs['memory']['type'](memory_size=self._memory_size, **kwargs['memory'])
        else:
            self._memory = ReplayMemory(memory_size=self._memory_size)
        self._eval_params = None
        # tensorflow tensors
        if isinstance(self._memory, PrioritizedReplayMemory):
            self._isw = tf.placeholder(dtype=tf.float32, shape=[None, ], name='importance_sampling_weights')
        self._state = tf.placeholder(dtype=tf.float32, shape=[None, self._state_dim], name='state')
        self._next_state = tf.placeholder(dtype=tf.float32, shape=[None, self._state_dim], name='next_state')
        self._reward = tf.placeholder(dtype=tf.float32, shape=[None, ], name='reward')
        self._action = tf.placeholder(dtype=tf.int32, shape=[None, ], name='action')
        self._loss = None
        self._eval_outs = None
        # tensorflow ops
        self._train_op = None

    def _build_model(self, **kwargs):
        """Builds the Deep Q-Network.

        Construct two equally structured Q-Networks (q_eval and q_target) according to child class' implementation.
        Defines _loss function, applies L2-Regulrization to _loss, defines optimizer and train operation"""
        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
            with tf.variable_scope("q_target", reuse=tf.AUTO_REUSE):
                target_outs = self._build_q_network(input_tensor=self._next_state, reuse=tf.AUTO_REUSE, **kwargs)
                target_params = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name)
                target_outs = tf.stop_gradient(target_outs)
            with tf.variable_scope("q_eval", reuse=tf.AUTO_REUSE):
                eval_outs = self._build_q_network(input_tensor=self._state, reuse=tf.AUTO_REUSE, **kwargs)
                eval_params = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name)
            with tf.variable_scope('replace', reuse=tf.AUTO_REUSE):
                self._replace_params = [tf.assign(t, e) for t, e in zip(target_params, eval_params)]
            with tf.variable_scope("merge", reuse=tf.AUTO_REUSE):
                _batch_size = tf.shape(self._state)[0]
                _range = tf.range(_batch_size)
                _a_indices = tf.stack([_range, self._action], axis=1)
                q_eval_actions = tf.gather_nd(params=eval_outs, indices=_a_indices)
                q_target_actions = self._reward + self._discount_factor * tf.reduce_max(target_outs, axis=1)
                self._td_error = tf.abs(tf.subtract(q_target_actions, q_eval_actions))
            with tf.variable_scope("loss", reuse=tf.AUTO_REUSE):
                if isinstance(self._memory, PrioritizedReplayMemory):
                    self._loss = self._isw * tf.square(self._td_error)
                else:
                    self._loss = tf.square(self._td_error)
                for var in eval_params:
                    self._loss = self._loss + self._l2_scale * tf.nn.l2_loss(var)
            with tf.variable_scope("opt", reuse=tf.AUTO_REUSE):
                optim = tf.train.RMSPropOptimizer(learning_rate=self._learning_rate)
                self._train_op = optim.minimize(self._loss, global_step=self._global_step, var_list=eval_params)
        # variable hooks
        self._eval_outs = eval_outs
        self._eval_params = eval_params

    @abstractmethod
    def _build_q_network(self, input_tensor, reuse=tf.AUTO_REUSE, **kwargs):
        """Builds a Q-Network for Q-Value estimation.

        Args:
            input_tensor (tf.Tensor): accordingly _state or _next_state tensor
            reuse (enum _ReuseMode): tensorflow's reusemode

        Returns:
            tf.Tensor: q_value output of Q-Network

        Raises:
            NotImplementedError: if abstract method is to be invoked
        """
        raise NotImplementedError("Must be implemented in subclass.")

    def get_params(self):
        """dict: returns the currently used variables in Q-eval Network"""
        var_dict = {}
        for var in self._eval_params:
            var_dict[var.name] = self._sess.run(var)
        return var_dict

    def act(self, state, **kwargs):
        """QAgent selects an action by either exploiting his knowledge or by exploring his action space. Exploitation
        depicts the inference on Q-eval with state, whereas Exploration depicts a uniformly drawn random action.

        Args:
            state (np.ndarray with shape=(_state_dim,)): input state

        Returns:
            int: action to be taken
        """
        if self._rnd_state.uniform(low=0.0, high=1.0) < self._epsilon:      # epsilon-greedy exploitation
            q_action = self._sess.run(self._eval_outs, feed_dict={self._state: [state]})
            action = np.argmax(q_action[0])
        else:                                                               # (1-epsilon) exploration
            action = self._rnd_state.randint(0, self._action_dim)
        return action

    def train(self, train_writer=None, **kwargs):
        """QAgent trains on a sample of replay memory.

        Firstly, requests a sample of size _sample_size from _memory. Secondly, batches through sample with _batch_size.
        Loss values for each batch are appended and returned.

        Args:
            train_writer (tf.summary.FileWriter): writer instance for tensorboard

        Returns:
            list of float: loss of training on drawn samples
        """
        # check for replace step
        if self._replace_threshold == self._replace_step:
            self._sess.run(self._replace_params)
            self._logger.info("replaced q_target's variables with q_eval's variables")
            self._replace_step = 0

        losses = []
        feed_dict = dict()
        # batch through sample
        for step in range(int(self._sample_size / self._batch_size)):

            # get train samples
            if isinstance(self._memory, PrioritizedReplayMemory):
                tree_indices, feed_dict[self._isw], experiences = self._memory.sample(sample_size=self._batch_size)
            else:
                experiences = self._memory.sample(sample_size=self._batch_size)

            # build feed dict
            feed_dict[self._state], feed_dict[self._action], feed_dict[self._reward], feed_dict[self._next_state] =\
                zip(*experiences)

            # train step
            _, _td_error, _loss = self._sess.run([self._train_op, self._td_error, self._loss], feed_dict=feed_dict)

            # save losses
            losses.append(_loss)

            # update priorities
            if isinstance(self._memory, PrioritizedReplayMemory):
                self._memory.update(tree_indices=tree_indices, td_errors=_td_error)
                self._memory.increment_beta()

        # replace step inc
        self._replace_step += 1
        return losses

    def store(self, state, action, reward, next_state):
        """Store quadruple of experience in replay memory.

        Args:
            state (np.ndarray with shape=(_state_dim,): observed initial state before action
            action (int): action selected for state
            reward (float): reward from environment for taking action in state
            next_state (np.ndarray with shape=(_state_dim,)): observed state after action
        """
        self._memory.store(experience=(state, action, reward, next_state))
