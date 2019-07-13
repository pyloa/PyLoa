import tensorflow as tf

from .qagent import QAgent
from .vertexagent import VertexAgent


class QMLPVertexAgent(QAgent, VertexAgent):
    """Agent with a 3-Layered-Multi-Layer-Perceptron as a Deep Q-Network.

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
        _units (int): number of units per layer
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 learning_rate,
                 units,
                 discount_factor,
                 memory_size,
                 sample_size,
                 batch_size,
                 epsilon,
                 epsilon_delta,
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
        """Builds a 3-Layer-Multi-Layer-Perceptron Q-Network for Q-Value estimation.

            Args:
                input_tensor (tf.Tensor): accordingly _state or _next_state tensor
                reuse (enum _ReuseMode): tensorflow's reusemode

            Returns:
                tf.Tensor: q_value output of Q-Network
        """
        with tf.variable_scope("vars", reuse=reuse):
            weight = tf.Variable(tf.random_normal([self._units//4, self._action_dim]), name='weight')
            bias = tf.Variable(tf.random_normal([self._action_dim]), name='bias')
        with tf.variable_scope("L1", reuse=reuse):
            L1 = tf.layers.dense(
                inputs=input_tensor,
                units=self._units,
                activation=tf.nn.leaky_relu,
                kernel_initializer=tf.random_normal_initializer(0., 0.3),
                bias_initializer=tf.constant_initializer(0.1),
                reuse=reuse,
            )
        with tf.variable_scope("L2", reuse=reuse):
            L2 = tf.layers.dense(
                inputs=L1,
                units=self._units//2,
                activation=tf.nn.leaky_relu,
                kernel_initializer=tf.random_normal_initializer(0., 0.3),
                bias_initializer=tf.constant_initializer(0.1),
                reuse=reuse,
            )
        with tf.variable_scope("L3", reuse=reuse):
            L3 = tf.layers.dense(
                inputs=L2,
                units=self._units//4,
                activation=tf.nn.leaky_relu,
                kernel_initializer=tf.random_normal_initializer(0., 0.3),
                bias_initializer=tf.constant_initializer(0.1),
                reuse=reuse,
            )
        with tf.variable_scope("weights_biases", reuse=reuse):
            outs = tf.add(tf.matmul(L3, weight), bias)
        return outs
