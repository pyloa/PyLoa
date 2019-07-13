from collections.abc import Iterable
import logging

import numpy as np


class SumTree(object):
    """A Fenwick (Sum)Tree implementation used as an indexing structure. This implementation assures numerical stability
    via auto-correction. This implementation is not threadsafe.

    SumTree can hold _memory_size many data entries, which are contained in _data as ring-buffer implementation. _tree
    is an array based implementation of a tree and depicts the SumTree used to index _data. Each leaf of _tree  indexes
    a data-entry instance. The value of each leaf in _tree depicts a data-entries' priority. Each non-leaf node in _tree
    is the sum of its children.

        Example:
            Assume a memory_size = 8 instance:

                _tree:
                                            24                          # sum of children's priorities
                                          /   \
                                         /     \
                                        /       \
                                       /         \
                                     13          11                     # sum of children's priorities
                                     / \         / \
                                    /   \       /   \
                                   4     9     5     6                  # sum of children's priorities
                                  / \   / \   / \   / \
                                 1  3  2  7  3  2  4  2                 # leafs; priority of each data entry below
                _data:
                                [a, b, c, d, e, f, g, h]                # coresponding data entries

                _tree as array:
                    [24, 13, 11, 4, 9, 5, 6, 1, 3, 2, 7, 3, 2, 4, 2]    # _tree as actual array


            To index _memory_size many data entries _tree needs to be of size: _memory_size * 2 - 1. A data_index in
            _data corresponds to the following tree_index:

                    tree_index = data_index + _memory_size - 1

                    <=>

                    data_index = tree_index - _memory_size + 1

            In this case the data_index of 'c' is 2; its tree_index is 2 + 8 - 1 = 9 and therefore its current priority
            is _tree[9] = 2. Updating the priority of an existing data entry is propagated upwards throughout _tree in
            O(log(2 * _memory_size - 1)). Assume invocation of update(tree_index=9, priority=1):

                _tree:
                                            23                      # sum of children's priorities
                                          /   \
                                         /     \
                                        /       \
                                       /         \
                                      12         11                 # sum of children's priorities
                                     / \         / \
                                    /   \       /   \
                                   4     8     5     6              # sum of children's priorities
                                  / \   / \   / \   / \
                                 1  3  1  7  3  2  4  2             # leafs; priority of each data entry below

            The update is, depending on the size of the tree and the range of _tree values, not always numerically
            stable. This means that _tree[0] = sum(_tree[-_memory_size:]) will eventually not hold to be true, if
            floating point precision is exceeded when relatively small priority updates are performed on a tree_index.

    Therefore, update is not always numerically stable; Priority updates within the index structure can result in small
    delta updates, which can not always be propagated upwards throughout the entire tree due to floating point precision
    limitations; the SumTree implementation at hand checks for "failing" delta updates and auto-corrects "too small
    delta changes" to the next applicable machine epsilon. If the auto-correct exceeds a relative change of
    _max_relative_change with said auto-correction, a warning is logged. SumTree is agnostic with regards to its data
    entries.

    Args:
        memory_size (int): number of data entries to store and index
        max_relative_change (number): maximum relative change of SumTree's auto-correction for numerical stability
        dtype (type): floating point precision used in SumTree

    Attributes:
        _memory_size (int): number of data entries to store and index
        _data_cursor (int): current cursor position
        _data (np.ndarray): container for data entries
        _tree_size (int): number of nodes needed to index _memory_size many data entries
        _max_relative_change (number): maximum relative change of SumTree's auto-correction for numerical stability
        _dtype (type): dtype for floating point precision used in SumTree
        _tree (np.ndarray): container for tree nodes
        _logger (logging.Logger): general purpose logger
    """
    def __init__(self, memory_size, max_relative_change, dtype):
        self._memory_size = memory_size
        self._data_cursor = 0
        self._data = np.zeros(shape=self._memory_size, dtype=object)
        self._tree_size = 2 * self._memory_size - 1
        self._max_relative_change = max_relative_change
        self._dtype = dtype
        self._tree = np.zeros(shape=(self._tree_size,), dtype=self._dtype)
        self._logger = logging.getLogger(__name__)
        self._logger.info("initializing {}".format(self.__class__.__name__))

    def __getitem__(self, index):
        """tuple(type(data)): data entries indexed in _tree with index"""
        if isinstance(index, Iterable):
            return tuple(map(self._get_item, index))
        return self._get_item(index),

    def __len__(self):
        """int: maximum number of data entries SumTree can hold"""
        return self._memory_size

    def add(self, priority, data_entry):
        """Places data_entry at cursor location in _data. Propagates new priority through tree. Increments cursor.
        Operation performs in  O(log(_memory_size)).
        
        Args:
            priority (number): initial priority value of data_entry
            data_entry (object): data_entry to index
        """
        _tree_index = self._data_cursor + len(self) - 1
        self._data[self._data_cursor] = data_entry
        self.update(_tree_index, priority)
        self._data_cursor = (self._data_cursor + 1) % len(self)

    def update(self, tree_index, priority):
        """Propagate a priority update for tree_index throughout tree. Operation performs in  O(log(_memory_size)).

        Calculates the _delta change to propagate through tree. If _delta can be applied to _tree[0], than this
        operation is numerically stable and no auto-correction is needed. If _delta can't be applied to _tree[0], due to
        _delta < np.spacing(_tree[0]), then _delta is auto-corrected to np.spacing(_tree[0]) (smallest possible machine
        epsilon to propagate change throughout SumTree). If _max_relative_change is exceeded, a warning is logged.

        Args:
            tree_index (int): index of leaf to update priority for
            priority (number): new priority

        Raises:
            AssertionError: tree_index is too small and therefore indexes a non-leaf node
        """
        # get max priority from tree, calculate delta change to propagate through SumTree
        assert tree_index >= len(self) - 1, "tree_index='{}' is not indexing a leaf".format(tree_index)
        _max_priority = self._tree[0]
        _delta = priority - self._tree[tree_index]

        # assure numerical stability, log warning if auto-correct is too excessive
        if _delta != 0.0 and _max_priority + _delta == _max_priority:       # floating precision exceeded
            _delta = np.sign(_delta) * np.spacing(_max_priority)            # minimum eps to apply change to tree
            _relative_change = _delta/(priority - self._tree[tree_index])   # relative change to _delta
            self._logger.debug("SumTree delta update changed from '{}' to '{}' (relative_change='{:5.5f}') due to "
                               "numerical stability".format(priority-self._tree[tree_index], _delta, _relative_change))
            if _relative_change > self._max_relative_change:                # relative change too excessive
                _warn_msg = "SumTree delta update propagation exceeds floating point precision for dtype='{}': " \
                            "normalize/clip td_errors and rewards OR allow max_relative_change='{}' to be larger"\
                    .format(self._dtype, self._max_relative_change)
                self._logger.warning(_warn_msg)

        # set current priority, propagate change through tree
        self._tree[tree_index] = priority
        while tree_index != 0 and _delta != 0.0:
            tree_index = (tree_index - 1) // 2
            self._tree[tree_index] += _delta

    def clear(self):
        """Clear containers and structures."""
        del self._data[:]
        self._data = np.zeros(shape=self._memory_size, dtype=object)
        self._data_cursor = 0
        self._tree.fill(0.0)

    def _get_priorities(self):
        """np.ndarray: priorities of data entries"""
        return self._tree[-self._memory_size:]

    def _get_total_priority(self):
        """number: prirority of root node (summed priority of all data entries)"""
        return self._tree[0]

    def _get_max_priority(self):
        """number: highest priority of a leaf node"""
        return np.max(self.priorities)

    def _get_item(self, item):
        """Helper function to retrieve an entry from SumTree with index item.

        Args:
            item (number): selector to index a single data entry with

        Returns:
            tuple of (int, number, type(data)): First item is the tree_index, at which selected entry is indexed with.
                Second item is the priority of corresponding the data entry. Third item is the corresponding data entry.
        """
        _parent = 0

        while _parent < self._memory_size - 1:
            _left_child = 2 * _parent + 1
            _right_child = _left_child + 1
            if item <= self._tree[_left_child]:
                _parent = _left_child
            else:
                item -= self._tree[_left_child]
                _parent = _right_child

        _data_index = _parent - len(self) + 1
        return _parent, self._tree[_parent], self._data[_data_index]

    def _get_data_size(self):
        """int: number of data entries stored"""
        return len(self._data[self._data != 0])

    data_size = property(fset=None, fget=_get_data_size, fdel=None)
    priorities = property(fset=None, fget=_get_priorities, fdel=None)
    max_priority = property(fset=None, fget=_get_max_priority, fdel=None)
    total_priority = property(fset=None, fget=_get_total_priority, fdel=None)
