import os
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from pyloa.utils.graph import Graph, NumpyGraph, EdgeListGraph, AdjacencyListGraph
from pyloa.utils.exception import DirectoryNotFoundError
from .instancegenerator import InstanceGenerator


class GraphGenerator(InstanceGenerator):
    """Abstract base class for to iteratively generate graph_number-many graphs of size graph_size.

    Args:
        graph_type (type): graph type (implementations of pyloa.utils.Graph) that self shall return
        graph_number (int): how many graph instances self will return before depletion
        graph_size (int): maximum size of graph (vertex count) that self will return

    Attributes:
        _graph_type (type): graph type (implementations of pyloa.utils.Graph) that self shall return
        _graph_size (int): maximum size of graph (vertex count) that self will return
        _vertices (set): vertices each graph has to contain

    Raises:
        AssertionError: if graph_number, graph_size does not exceed 0 or if graph_type is not of appropriate type
        NotImplementedError: if abstract class is to be instanced
    """
    def __init__(self,
                 graph_type,
                 graph_number,
                 graph_size):
        assert graph_number > 0, "graph_number must exceed 0 but received {} instead".format(graph_number)
        assert graph_size > 0, "graph_size must exceed 0 but received {} instead".format(graph_size)
        assert issubclass(graph_type, Graph), "graph_type must be subclass of {}".format(Graph)
        super(GraphGenerator, self).__init__(instance_number=graph_number)
        self._graph_type = graph_type
        self._graph_size = graph_size
        self._vertices = {v for v in range(self._graph_size)}

    def _get_vertices(self):
        """set of int: vertices each generated graph instance will contain"""
        return deepcopy(self._vertices)

    def _get_graph_size(self):
        """int: number of vertices each generated graph instance will consist of"""
        return self._graph_size

    def _get_graph_type(self):
        """type: graph_type that self generates"""
        return self._graph_type

    def _set_graph_type(self, value):
        """Set _graph_type.

        Args:
            value (type): graph_type that self generates

        Raises:
            AssertionError: if value is not appropiate type (subclass implementation of Graph)
        """
        assert isinstance(value, Graph), "graph_type must subclass {} but found {} instead".format(Graph, type(value))
        self._graph_type = value

    vertices = property(fset=None, fget=_get_vertices, fdel=None)
    graph_size = property(fset=None, fget=_get_graph_size, fdel=None)
    graph_type = property(fset=_set_graph_type, fget=_get_graph_type, fdel=None)


class RandomGraphGenerator(GraphGenerator):
    """A GraphGenerator that generates graph_number-many random NumpyGraphs of size graph_size. The initial coloring
    of each graph instance is undefined (all vertices have color 0).

    Args:
        graph_number (int): how many graph instances self will return before depletion
        graph_size (int): exact size of graph (vertex count) that self will return
        dtype (type): dtype used for NumpyGraph's adjacency_matrix
        seed (int): seed for random number generation

    Attributes:
        _seed (int or None): seed _rnd_state was initialized with
        _rnd_state (np.random.RandomState): random number generator for self with own local seed
        _dtype (type): dtype used for NumpyGraph's adjacency_matrix
    """
    def __init__(self,
                 graph_number,
                 graph_size,
                 dtype=np.uint8,
                 seed=None,
                 **kwargs):
        super(RandomGraphGenerator, self).__init__(
            graph_type=NumpyGraph,
            graph_number=graph_number,
            graph_size=graph_size,
        )
        self._rnd_state = np.random.RandomState(seed=seed)
        self._seed = seed
        self._dtype = dtype
        self._logger.info("initialized {}".format(self))

    def _generate_instance(self):
        """NumpyGraph: newly generated graph instance"""
        # randomly roll a matrix
        rnd_matrix = self._rnd_state.randint(0, 2, size=(self._graph_size, self._graph_size), dtype=self._dtype)
        # use upper triangle to enforce direction and set diagonal to 0
        rnd_matrix -= np.tril(rnd_matrix, 0)
        rnd_matrix += np.triu(rnd_matrix, 1).transpose()
        return self.graph_type(adjacency_matrix=rnd_matrix, vertex_colors=np.zeros(self._graph_size, dtype=self._dtype))

    def __str__(self):
        """str: string representation of self"""
        return self.__class__.__name__ + "(seed={})".format(self._seed)


class DeltaRandomGraphGenerator(GraphGenerator):
    """A GraphGenerator that generates graph_number-many random NumpyGraphs of size graph_size where for each vertex of
    a generated graph instance degree(vertex) <= max_delta holds to be true. The initial coloring of each graph instance
    is undefined (all vertices have color 0).

    Args:
        max_delta (int): maximum degree that each vertex of generated graphs are allowed to have
        graph_number (int): how many graph instances self will return before depletion
        graph_size (int): exact size of graph (vertex count) that self will return
        dtype (type): dtype used for NumpyGraph's adjacency_matrix
        seed (int): seed for random number generation

    Attributes:
        _max_delta (int): maximum degree that each vertex of generated graphs are allowed to have
        _seed (int or None): seed _rnd_state was initialized with
        _rnd_state (np.random.RandomState): random number generator for self with own local seed
        _dtype (type): dtype used for NumpyGraph's adjacency_matrix
    """
    def __init__(self,
                 max_delta,
                 graph_number,
                 graph_size,
                 dtype=np.uint8,
                 seed=None,
                 **kwargs):
        assert max_delta >= 2, "max_delta must exceed 1 but received {} instead".format(max_delta)
        super().__init__(
            graph_type=NumpyGraph,
            graph_number=graph_number,
            graph_size=graph_size,
        )
        self._max_delta = max_delta
        self._rnd_state = np.random.RandomState(seed=seed)
        self._seed = seed
        self._dtype = dtype
        self._heatmap = np.zeros(shape=(self._graph_size, self._graph_size), dtype=np.float64)
        self._count = 0
        self._logger.info("initialized {}".format(self))

    def _generate_instance(self):
        """NumpyGraph: newly generated graph instance"""
        adjacency_matrix = np.zeros(shape=(self._graph_size, self._graph_size), dtype=self._dtype)
        degrees = self._rnd_state.randint(low=0, high=self._max_delta + 1, size=self.graph_size)

        _amax = np.amax(degrees)
        arg_maxes = np.argwhere(degrees == _amax).flatten()
        idx = arg_maxes[self._rnd_state.randint(len(arg_maxes))]
        max_degree = degrees[idx]
        if max_degree == 0:
            self._update_heatmap(adjacency_matrix)
            return self.graph_type(adjacency_matrix=adjacency_matrix,
                                   vertex_colors=np.zeros(self._graph_size, self._dtype))
        indices = np.nonzero(degrees)

        index = np.argwhere(indices == idx)[0][1]
        new_indices = np.delete(indices, index)

        length = len(new_indices)
        max_degree = np.min([max_degree, length])

        while max_degree > 0:
            choice = self._rnd_state.choice(new_indices, size=max_degree, replace=False)
            for k in choice:
                adjacency_matrix[idx][k], adjacency_matrix[k][idx] = 1, 1
                degrees[k] = degrees[k] - 1
            degrees[idx] = 0
            _amax = np.amax(degrees)
            arg_maxes = np.argwhere(degrees == _amax).flatten()
            idx = arg_maxes[self._rnd_state.randint(len(arg_maxes))]
            max_degree = degrees[idx]
            if max_degree == 0:
                break
            indices = np.nonzero(degrees)
            index = np.argwhere(indices == idx)[0][1]
            new_indices = np.delete(indices, index)
            length = len(new_indices)
            max_degree = np.min([max_degree, length])
        self._update_heatmap(adjacency_matrix)
        return self.graph_type(adjacency_matrix=adjacency_matrix, vertex_colors=np.zeros(self._graph_size, self._dtype))

    def __str__(self):
        """str: string representation of self"""
        return self.__class__.__name__ + "(delta={}, seed={})".format(self._max_delta, self._seed)

    def _get_max_delta(self):
        """int: returns maximum degree that each vertex of a generated graph instance is allowed to have"""
        return self._max_delta

    def _update_heatmap(self, matrix):
        """updates the heatmap of the generator"""
        _heatmap = self._heatmap * self._count + matrix
        self._count += 1
        self._heatmap = _heatmap / self._count

    def _get_heatmap(self):
        """np.ndarray: the heatmap of the generator"""
        return self._heatmap

    def save_heatmap(self, filepath, min_value=0.0, max_value=1.0):
        """Plots the heatmap and stores it at filepath.

        Attributes:
            filepath (str): filepath of plot to be stored at
            min_value (float): minimum value for heatmap (coldest value = blue)
            max_value (float): maximum value for heatmap (warmest value = yellow)

        Raises:
            DirectoryNotFoundError: If filepath can't be saved, since parent directories (the path) doesn't exist
        """
        if not os.path.exists(os.path.dirname(filepath)):
            raise DirectoryNotFoundError("directory '{}' does not exist".format(os.path.dirname(filepath)))
        plt.imshow(self.heatmap, cmap='viridis', vmin=min_value, vmax=max_value)
        plt.colorbar()
        plt.savefig(filepath)
        plt.close()

    max_delta = property(fset=None, fget=_get_max_delta, fdel=None)
    heatmap = property(fset=None, fget=_get_heatmap, fdel=None)
