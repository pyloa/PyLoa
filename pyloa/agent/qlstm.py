import tensorflow as tf

from .qagent import QAgent
from .pagingagent import PagingAgent


class QLSTMAgent(QAgent, PagingAgent):
    """Agent with a single LSTM-Cell as a Deep Q-Network.

    Args:
        state_dim (int): defines input shape=(_state_dim,) of input states
        action_dim (int): defines the number of different actions the agent can take
        learning_rate (float): learning_rate of agent's model
        units (int): number of units per layer
        discount_factor (float): value at which future Q-target Values are discounted
        memory_size (int): total size of agent's replay memory
        sample_size (int): size of sample randomly drawn from memory to train on per train step
        batch_size (int): size of each batch a sample for training is batched with
        epsilon (float): agent's current chance to exploit his knowledge, else explore action_space
        epsilon_delta (float): increment value of epsilon for each increment call
        epsilon_max (float): maximum value of epsilon
        l2_scale (float): L2-Regularization scale for eval network
        replace_threshold (int): depicts after how many train steps Q-target's variables are replaced with Q-eval's
        save_file (str): location at which agent's checkpoints are stored at

    Attributes:
        _units (int): number of units for LSTM Cell
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 learning_rate,
                 units,
                 discount_factor,
                 memory_size=5000,
                 sample_size=1000,
                 batch_size=32,
                 epsilon=0.90,
                 epsilon_delta=50e-4,
                 epsilon_max=1.0,
                 l2_scale=0.002,
                 replace_threshold=25,
                 save_file=None,
                 **kwargs):
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            memory_size=memory_size,
            sample_size=sample_size,
            batch_size=batch_size,
            epsilon=epsilon,
            epsilon_delta=epsilon_delta,
            epsilon_max=epsilon_max,
            l2_scale=l2_scale,
            replace_threshold=replace_threshold,
            save_file=save_file,
            **kwargs
        )
        # params
        self._units = units
        # init
        self._build_model(**kwargs)
        self._sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

    # noinspection PyMethodOverriding
    def _build_q_network(self, input_tensor, reuse=tf.AUTO_REUSE, **kwargs):
        """Builds a LSTM Q-Network for Q-Value estimation.

            Args:
                input_tensor (tf.Tensor): accordingly _state or _next_state tensor
                reuse (enum _ReuseMode): tensorflow's reusemode

            Returns:
                tf.Tensor: q_value output of Q-Network
        """
        with tf.variable_scope("vars", reuse=reuse):
            weight = tf.Variable(tf.random_normal([self._units, self._action_dim]), name='weight')
            bias = tf.Variable(tf.random_normal([self._action_dim]), name='bias')
        with tf.variable_scope("cell", reuse=reuse):
            cell = tf.nn.rnn_cell.LSTMCell(
                num_units=self._units,
                forget_bias=1.0,
                state_is_tuple=True,
                activation=tf.nn.leaky_relu,
                dtype=tf.float32,
                reuse=reuse,
            )
        with tf.variable_scope("rnn", reuse=reuse):
            outs, _ = tf.nn.static_rnn(
                cell=cell,
                inputs=[input_tensor],
                dtype=tf.float32
            )
            outs = tf.add(tf.matmul(outs[-1], weight), bias)
        return outs
