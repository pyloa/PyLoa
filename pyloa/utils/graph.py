from abc import abstractmethod
import numpy as np
from collections import OrderedDict
from copy import deepcopy


class Graph(object):
    def __init__(self, vertex_colors):
        if type(self) == Graph:
            raise NotImplementedError("{} must be subclassed.".format(self.__class__.__name__))
        self._vertex_colors = vertex_colors.astype(dtype=np.uint8)

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError("Must be implemented in subclass.")

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError("Must be implemented in subclass.")

    @abstractmethod
    def __str__(self):
        raise NotImplementedError("Must be implemented in subclass.")

    def __len__(self):
        return len(self._vertex_colors)

    def __iter__(self):
        return self.__next__()

    def __next__(self):
        for vertex in range(len(self)):
            yield vertex

    def _get_predecessors(self, vertex):
        return self[:, vertex]

    def _get_successors(self, vertex):
        return self[vertex]

    def _get_vertex_colors(self):
        return self._vertex_colors

    def _get_vertices(self):
        return set(range(len(self)))

    @abstractmethod
    def add_edge(self, edge):
        raise NotImplementedError("Must be implemented in subclass.")

    @abstractmethod
    def has_edge(self, edge):
        raise NotImplementedError("Must be implemented in subclass.")

    @abstractmethod
    def remove_edge(self, edge):
        raise NotImplementedError("Must be implemented in subclass.")

    @abstractmethod
    def copy(self):
        raise NotImplementedError("Must be implemented in subclass.")

    def get_color(self, vertex):
        return self._vertex_colors[vertex]

    def get_neighbours(self, vertex):
        return self._get_successors(vertex)

    def get_neighbours_colors(self, vertex):
        _neighbour_colors = self._vertex_colors[self[vertex]]
        return _neighbour_colors[_neighbour_colors > 0]

    def get_degree(self, vertex):
        return len(self[vertex])

    def get_max_degree(self):
        return max([self.get_degree(vertex) for vertex in self])

    def is_locally_colorized(self, vertex):
        return self.get_color(vertex) not in self.get_neighbours_colors(vertex)

    def is_colorized(self):
        if 0 in self.coloring:
            return False
        for vertex in self:
            if not self.is_locally_colorized(vertex):
                return False
        return True

    def astype(self, graph_type, dtype=np.uint8):
        if graph_type == NumpyGraph:
            adjacency_matrix = np.zeros(shape=(len(self), len(self)), dtype=dtype)
            for vertex in self:
                adjacency_matrix[vertex][self[vertex]] = 1
            return NumpyGraph(
                adjacency_matrix=adjacency_matrix,
                vertex_colors=self._vertex_colors,
                dtype=dtype
            )
        if graph_type == AdjacencyListGraph:
            dictionary = OrderedDict()
            for vertex in self:
                dictionary[vertex] = list(self[vertex])
            return AdjacencyListGraph(
                dictionary=dictionary,
                vertex_colors=self._vertex_colors
            )
        if graph_type == EdgeListGraph:
            edge_list = []
            for vertex in self:
                for n_vertex in self[vertex]:
                    edge = {vertex, n_vertex}
                    if edge not in edge_list:
                        edge_list.append(edge)
            return EdgeListGraph(
                edge_list=edge_list,
                vertex_colors=self._vertex_colors
            )

    def __eq__(self, other):
        # correct class, correct number of vertices
        if not isinstance(other, Graph) or len(self) != len(other):
            return False
        for vertex in self:
            # vertex equality
            if vertex not in other or self.get_color(vertex) != other.get_color(vertex):
                return False
            # neighbour equality
            for n_vertex in self[vertex]:
                if not other.has_edge({vertex, n_vertex}):
                    return False
        return True

    number_of_vertices = property(fset=None, fget=__len__, fdel=None)
    coloring = property(fset=None, fget=_get_vertex_colors, fdel=None)
    vertices = property(fset=None, fget=_get_vertices, fdel=None)


class AdjacencyListGraph(Graph):
    def __init__(self, dictionary, vertex_colors):
        super(AdjacencyListGraph, self).__init__(vertex_colors=vertex_colors)
        self._dictionary = deepcopy(dictionary)

    def __getitem__(self, index):
        return self._dictionary[index]

    def __repr__(self):
        return repr(self._dictionary)

    def __str__(self):
        return str(self._dictionary)

    def add_edge(self, edge):
        v1, v2 = edge
        self[v1].append(v2)
        self[v2].append(v1)

    def has_edge(self, edge):
        v1, v2 = edge
        return v1 in self[v2] and v2 in self[v1]

    def remove_edge(self, edge):
        v1, v2 = edge
        self[v1].remove(v2)
        self[v2].remove(v1)

    def copy(self):
        return AdjacencyListGraph(dictionary=self._dictionary, vertex_colors=self._vertex_colors)


class EdgeListGraph(Graph):
    def __init__(self, edge_list, vertex_colors):
        super(EdgeListGraph, self).__init__(vertex_colors=vertex_colors)
        self._edge_list = deepcopy(edge_list)

    def __getitem__(self, index):
        return [edge.difference({index}).pop() for edge in self._edge_list if index in edge]

    def __repr__(self):
        return repr(self._edge_list)

    def __str__(self):
        return str(self._edge_list)

    def add_edge(self, edge):
        self._edge_list.append(set(edge))

    def has_edge(self, edge):
        return set(edge) in self._edge_list

    def remove_edge(self, edge):
        self._edge_list.remove(set(edge))

    def copy(self):
        return EdgeListGraph(edge_list=self._edge_list, vertex_colors=self._vertex_colors)


class NumpyGraph(Graph):
    def __init__(self, adjacency_matrix, vertex_colors, dtype=np.uint8):
        super(NumpyGraph, self).__init__(vertex_colors=vertex_colors)
        self._adjacency_matrix = adjacency_matrix.astype(dtype=dtype)
        self._dtype = dtype

    def __getitem__(self, index):
        return np.flatnonzero(self._adjacency_matrix[index])

    def __repr__(self):
        return repr(self._adjacency_matrix)

    def __str__(self):
        return str(self._adjacency_matrix)

    def add_edge(self, edge):
        v1, v2 = edge
        self._adjacency_matrix[([v1, v2], [v2, v1])] = 1

    def has_edge(self, edge):
        v1, v2 = edge
        return self._adjacency_matrix[v1, v2] == self._adjacency_matrix[v2, v1] == 1

    def remove_edge(self, edge):
        v1, v2 = edge
        self._adjacency_matrix[([v1, v2], [v2, v1])] = 0

    def copy(self):
        return NumpyGraph(adjacency_matrix=self._adjacency_matrix, vertex_colors=self._vertex_colors, dtype=self._dtype)
