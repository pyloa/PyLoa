import numpy as np

from .environment import Environment
from pyloa.utils.graph import Graph


class VertexEnvironment(Environment):
    """Base class for pyloa.agent.VertexAgent agents to play on.

    Args:
        max_delta (int): maximum degree in graph that a vertex can have
        max_color (int): number of colors that graph is allowed to be colored with
        include_vertex (bool): depicts whether or not the current vertex is included within state
        include_delta (bool): depicts whether or not first order neighborhood colors are included within state
        include_delta_squared (bool): depicts whether or not second order neighborhood colors are included within state
        include_color_flag (bool): depicts whether or not _color flags is included within state
        include_num_steps (bool): depicts whether or not _num_steps is included within state
        include_embedding (bool): depicts whether or not embedding is included within state
        include_vertex_coloring (bool): depicts whether or not current graph coloring is included within state
        delta_padding_value (int): default value for delta and delta² if vertex has less than max_delta neighbors

    Attributes:
        _max_delta (int): maximum degree in graph that a vertiex can have
        _max_color (int): number of colors that graph is allowed to be colored with
        _include_vertex (bool): depicts whether or not the current vertex is included within state
        _include_delta (bool): depicts whether or not first order neighborhood colors are included within state
        _include_delta_squared (bool): depicts whether or not second order neighborhood colors are included within state
        _include_color_flag (bool): depicts whether or not _color flags is included within state
        _include_num_steps (bool): depicts whether or not _num_steps is included within state
        _include_embedding (bool): depicts whether or not embedding is included within state
        _include_vertex_coloring (bool): depicts whether or not current graph coloring is included within state
        _delta_padding_value (int): default value for delta and delta² if vertex has less than max_delta neighbors
        _last_state (dict or np.ndarray): the previous state of self
        _graph (pyloa.utils.Graph): the graph instance agents plas on (tries to colorize)
        _vertex_order (list of int): the order, in which environment will iterate over graph
        _num_steps (int): number of steps already played on current graph instance
        _color_flags (np.npdarray, shape=(_max_color,), dtype=np.int32): depicts if color at index has been used yet

    Raises:
        NotImplementedError: if abstract class is to be instanced
    """
    def __init__(self,
                 max_delta,
                 max_color,
                 include_vertex=True,
                 include_delta=True,
                 include_delta_squared=False,
                 include_color_flag=True,
                 include_num_steps=False,
                 include_embedding=False,
                 include_vertex_coloring=False,
                 delta_padding_value=-1,
                 **kwargs):
        super(VertexEnvironment, self).__init__()
        assert max_color >= max_delta
        # params
        self._max_delta = max_delta
        self._max_color = max_color
        self._delta_padding_value = delta_padding_value
        # state config
        self._include_vertex = include_vertex
        self._include_delta = include_delta
        self._include_delta_squared = include_delta_squared
        self._include_color_flag = include_color_flag
        self._include_num_steps = include_num_steps
        self._include_embedding = include_embedding
        self._include_vertex_coloring = include_vertex_coloring
        # structures
        self._last_state = None
        self._graph = None
        self._vertex_order = list(range(max_delta))
        self._num_steps = 0
        self._color_flags = np.zeros(self._max_color, dtype=np.int32)

    def _get_embedding(self):
        """np.ndarray: global embedding of the graph"""
        raise NotImplementedError("not implemented yet")

    def _is_terminated(self):
        """bool: True, if environment is terminated (all requests for _sequence evaluated). False otherwise."""
        return self._num_steps > 0 and self._vertex_order is not None and self._num_steps == len(self._vertex_order)

    def _is_colorized(self):
        """bool: True, if graph is correctly colorized. False otherwise."""
        return self._graph.is_colorized()

    def _is_locally_colorized(self):
        """bool: True, if graph at self.vertex is correctly locally colorized. False otherwise."""
        return self._graph.is_locally_colorized(self.vertex)

    def _was_locally_colorized(self):
        """bool: True, if graph at self.last_vertex is correctly locally colorized. False otherwise."""
        return self._graph.is_locally_colorized(self.last_vertex)

    def _get_action_dim(self):
        """int: number of actions"""
        return self._max_color

    def _get_state_dim(self):
        """int: length of state vector"""
        _sum = 0
        if self._include_vertex:
            _sum += 1
        if self._include_num_steps:
            _sum += 1
        if self._include_delta:
            _sum += self._max_delta
        if self._include_delta_squared:
            _sum += self._max_delta ** 2
        if self._include_color_flag:
            _sum += len(self._color_flags)
        if self._include_embedding:
            raise NotImplementedError("no embedding implementation yet in graph")
        if self._include_vertex_coloring and self._graph is not None:
            _sum += len(self._graph.coloring)
        return _sum

    def _get_delta(self):
        """np.ndarray: returns the first order neighborhood colors of self.vertex"""
        _neighbour_colors = self._graph.get_neighbours_colors(self.vertex)
        _res = np.ones(self._max_delta, dtype=np.int32) * self._delta_padding_value
        _res[0:len(_neighbour_colors)] = _neighbour_colors
        return _res

    def _get_delta_squared(self):
        """np.ndarray: returns the second order neighborhood colors of self.vertex"""
        _neighbours = self._graph[self.vertex]
        _res = np.empty(0, dtype=np.int32)
        for idx in range(self._max_delta):
            _tmp = np.ones(self._max_delta, dtype=np.int32) * self._delta_padding_value
            if idx < len(_neighbours):
                _neighbour_colors = self._graph.get_neighbours_colors(_neighbours[idx])
                _tmp[0:len(_neighbour_colors)] = _neighbour_colors
            _res = np.append(_res, _tmp)
        return _res

    def _get_instance(self):
        """Graph: returns the currently played on graph instance"""
        return self._graph

    def _get_num_colors(self):
        """int: returns the number of currently used colors on current graph instance"""
        return np.sum(self._color_flags)

    def _get_num_steps(self):
        """int: returns the number of played steps on current graph instance"""
        return self._num_steps

    def _get_current_vertex(self):
        """int: returns the current vertex to colotize for the agent"""
        return self._vertex_order[min(self._num_steps, len(self._vertex_order)-1)]

    def _get_last_vertex(self):
        """int: returns the previous vertex that has been colored with the last step"""
        if self.is_terminated:
            return self._vertex_order[-1]
        return self._vertex_order[min(self._num_steps, len(self._vertex_order)-1)-1]

    def _set_instance(self, value):
        """Overrides currently stored _graph with value without invoking reset()

        Args:
            value (pyloa.utils.Graph): new graph instance to play on

        Raises:
            AssertionError: if value is not a Graph instance
        """
        assert isinstance(value, Graph)
        self._graph = value.copy()
        self._vertex_order = [v for v in self._graph]
        self._num_steps = 0
        self._color_flags[np.flatnonzero(self._graph.coloring)] = 1

    def get_state(self, vectorized_state=False, vectorized_dtype=np.int32):
        """Returns the state of environment. The state is either vectorized (returns np.ndarray) or not (returns dict).
        The state comprises copies of each variable that has been defined to be included during environment creation.

        Args:
            vectorized_state: defines return type. A vectorized_state will be of type np.ndarray, otherwise dict
            vectorized_dtype: dtype of np.darray for vectorized_state

        Returns:
            np.ndarray or dict: state depicting the current observation (changing the returned state won't change the
                internal structures in environment)
        """
        if self.is_terminated:
            if vectorized_state:
                return np.ones_like(self._last_state) * (-2)
            else:
                return dict()

        if vectorized_state:
            _state = np.empty(0, dtype=vectorized_dtype)
            if self._include_vertex:
                _state = np.append(_state, np.asarray(self.vertex, dtype=vectorized_dtype))
            if self._include_num_steps:
                _state = np.append(_state, np.asarray(self._num_steps, dtype=vectorized_dtype))
            if self._include_delta:
                _state = np.append(_state, self.delta)
            if self._include_delta_squared:
                _state = np.append(_state, self.delta_squared)
            if self._include_color_flag:
                _state = np.append(_state, self._color_flags)
            if self._include_embedding:
                _state = np.append(_state, np.asarray(self.embedding, dtype=vectorized_dtype))
            if self._include_vertex_coloring:
                _state = np.append(_state, self._graph.coloring)
        else:
            _state = dict()
            _state['current_vertex'] = self.vertex
            _state['num_steps'] = self._num_steps
            _state['delta'] = self.delta
            _state['delta_squared'] = self.delta_squared
            _state['color_flag'] = np.copy(self._color_flags)
            _state['embedding'] = self._graph
            _state['vertex_coloring'] = np.copy(self._graph.coloring)
        self._last_state = _state
        return _state

    def calculate_state_complexity(self):
        raise NotImplementedError("not implemented yet")

    def step(self, action, vectorized_state=True):
        """Environment computes the step for action, updates all internal structures and returns the next state and cost
        as a tuple.

        Args:
            action (int): depicts the action (color) that the agent wants color self.vertex with
            vectorized_state (bool): depicts the return type of the observation after step/action is taken

        Returns:
            tuple: first entry is the next state, second entry are the costs affiliated with last action in last state

        Raises:
            AssertionError: if action is not in interval [0, max_color-1] and therefore invalid
        """
        assert 0 <= action < self._max_color, "action not in bounds: {}".format(action)
        picked_color = action + 1
        neighbour_colors = self._graph.get_neighbours_colors(self.vertex)
        if picked_color in neighbour_colors:
            cost = 1000
        elif self._color_flags[action] == 0:
            cost = 250
        else:
            cost = 0
        self._graph.coloring[self.vertex] = picked_color
        self._color_flags[action] = 1
        self._num_steps += 1
        return self.get_state(vectorized_state=vectorized_state), cost

    def reset(self):
        """Resets the environment to its initial state."""
        self._graph = None
        self._last_state = None
        self._num_steps = 0
        self._vertex_order.clear()
        self._color_flags[:] = 0

    vertex = property(fset=None, fget=_get_current_vertex, fdel=None)
    last_vertex = property(fset=None, fget=_get_last_vertex, fdel=None)
    embedding = property(fset=None, fget=_get_embedding, fdel=None)
    delta = property(fset=None, fget=_get_delta, fdel=None)
    delta_squared = property(fset=None, fget=_get_delta_squared, fdel=None)
    instance = property(fset=_set_instance, fget=_get_instance, fdel=None)
    is_terminated = property(fset=None, fget=_is_terminated, fdel=None)
    is_locally_colorized = property(fset=None, fget=_is_locally_colorized, fdel=None)
    was_locally_colorized = property(fset=None, fget=_was_locally_colorized, fdel=None)
    is_colorized = property(fset=None, fget=_is_colorized, fdel=None)
    num_colors = property(fset=None, fget=_get_num_colors, fdel=None)
    action_dim = property(fset=None, fget=_get_action_dim, fdel=None)
    state_dim = property(fset=None, fget=_get_state_dim, fdel=None)
    num_steps = property(fset=None, fget=_get_num_steps, fdel=None)
