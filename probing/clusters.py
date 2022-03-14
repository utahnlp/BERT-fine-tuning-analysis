# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Yichu Zhou - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.6.0
#
# Date: 2020-03-20 10:56:58
# Last modified: 2021-01-07 10:15:13

"""
Data structure for probing.
"""

# import logging
from collections import Counter
from functools import total_ordering
import heapq
from typing import List
# from multiprocessing import Pool

# import numpy as np
# import torch
# from tqdm import tqdm
# from tqdm import trange
# from joblib import Parallel, delayed


class Cluster:
    # __slots__ is used here because there will be
    # so many Cluster object during the probing, I
    # want to save as much memory as possible.
    __slots__ = ('indices', 'major_label',
                 '_hash_value',
                 'children', 'labels')

    def __init__(self, indices: List[int], labels: List[int]):
        """Initialize a new cluster with indices

        Args:
            - indices: The index of each point.
            - labels: The label of each point.
        """
        assert len(indices) == len(labels)
        self.indices = sorted(indices)
        self.labels = labels
        self.major_label = Counter(labels).most_common(1)[0][0]

        self._hash_value = ' '.join([str(i) for i in self.indices])
        self._hash_value = hash(self._hash_value)

        # The children is used to track the path of merging
        # This can be used to speed up the probing during later steps.
        self.children = set()

    @property
    def purity(self) -> float:
        n = sum([1 for i in self.labels if i == self.major_label])
        return n / len(self.labels)

    @staticmethod
    def merge(A: 'Cluster', B: 'Cluster') -> 'Cluster':
        """Merge two clusters and produce a new cluster.
        """
        assert type(A) == Cluster
        assert type(B) == Cluster
        indices = A.indices + B.indices
        labels = A.labels + B.labels
        reval = Cluster(indices, labels)
        reval.children = A.children | B.children

        # Do not forget A and B themselves.
        reval.children.add(A)
        reval.children.add(B)
        return reval

    def __hash__(self):
        return self._hash_value

    def __eq__(self, other):
        return self._hash_value == other._hash_value

    def __repr__(self):
        n = len(self.indices)
        idx = ' '.join([str(self.indices[i]) for i in range(n)])
        labels = ' '.join([str(self.labels[i]) for i in range(n)])
        s = 'Cluster(Indices:{a}, labels:{b}, major_label={c}, purity={d})'
        s = s.format(a=str(idx), b=labels, c=self.major_label, d=self.purity)
        return s


@total_ordering
class ClusterDisPair:
    """This is a intermediate class which is used to compute
    the distance between two clusters.
    This class should not be exposed to the end user.
    """
    __slots__ = ('i', 'j', 'dis', '_hash_value')

    def __init__(self, i: int, j: int, dis: float):
        """
        Be note,  here the index i and index j is not the index of
        points, instead they are the indices of clusters.

        Args:
            i: The index of the cluster.
            j: The index of the cluster.
            dis: The distance between these two clusters.
        """
        assert i != j
        self.i = min(i, j)
        self.j = max(i, j)
        self.dis = dis
        self._hash_value = hash((self.i, self.j, dis))

    def __hash__(self):
        return self._hash_value

    def __eq__(self, other):
        return self._hash_value == other._hash_value

    def __lt__(self, other):
        return self.dis < other.dis

    def __le__(self, other):
        return self.dis <= other.dis

    def __repr__(self):
        s = 'ClusterDisPair(i:{a}, j:{b}, dis:{c})'
        s = s.format(a=str(self.i), b=str(self.j), c=str(self.dis))
        return s


@total_ordering
class ClusterDisList:
    """ A heap list of pair of clusters.

    Each list represents all the pair distance of (idx,  i),  i < idx.
    Here,  idx and i are the indices of clusters.
    """
    __slots__ = ('dis_list', 'idx', '_hash_value')

    def __init__(self, dis_list: List[ClusterDisPair], idx: int):
        self.dis_list = dis_list
        heapq.heapify(self.dis_list)
        self.idx = idx
        self._hash_value = hash(idx)

    def min(self) -> ClusterDisPair:
        """Return the pair of minimum distance of this list.
        """
        return heapq.heappop(self.dis_list)

    def deactive(self):
        self.dis_list = []

    def __hash__(self):
        return self._hash_value

    def __eq__(self, other):
        if not self.dis_list and not other.dis_list:
            return True
        elif not self.dis_list or not other.dis_list:
            return False
        else:
            return self.dis_list[0] == other.dis_list[0]

    def __lt__(self, other):
        if not self.dis_list:
            return False
        if not other.dis_list:
            return True
        return self.dis_list[0] < other.dis_list[0]

    def __le__(self, other):
        if not self.dis_list:
            return False
        if not other.dis_list:
            return True
        return self.dis_list[0] <= other.dis_list[0]

    def __repr__(self):
        if not self.dis_list:
            s = 'Index:{a} is deactivate'
            s = s.format(a=str(self.idx))
        else:
            s = 'Index:{a} has minimum value {b}'
            s = s.format(a=str(self.idx),
                         b=str(self.dis_list[0]))
        return s


if __name__ == '__main__':
    import random
    n = 3
    array = []
    for i in range(n):
        for j in range(n):
            array.append(ClusterDisPair(i, j, i+j))
    random.shuffle(array)
    print(array)
    array.sort()
    print(array)
