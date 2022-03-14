# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Yichu Zhou - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.6.0
#
# Date: 2019-07-24 09:55:34
# Last modified: 2021-03-03 09:44:37

"""
Some utility functions, including loading and saving data.
"""

import collections
from typing import List, Tuple, TextIO, Dict

import numpy as np
import torch

from probing.distanceQ import DistanceQ


Pair = collections.namedtuple('Pair', ['Entity', 'Label'])


def load_entities(path: TextIO):
    reval = []
    with open(path, encoding='utf8') as f:
        for line in f:
            s = line.strip().split('\t')
            reval.append(Pair(*s))
    return reval


def load_labels(path: TextIO):
    reval = set()
    with open(path, encoding='utf8') as f:
        for line in f:
            reval.add(line.strip())
    return reval


def load_embeddings(path: TextIO):
    reval = []
    with open(path, encoding='utf8') as f:
        for line in f:
            s = line.strip().split()
            vec = [float(v) for v in s]
            reval.append(vec)
    return reval


def write_predictions(
        path: TextIO,
        cluster_list: List[List[Tuple[int, str, float]]],
        real_labels: List[str]):
    """ Write distances to the file.
    """
    assert len(cluster_list) == len(real_labels)
    with open(path, 'w', encoding='utf8') as f:
        for i, label_dis_pair_list in enumerate(cluster_list):
            line = [str(real_labels[i])]
            for cls_id, label, dis in label_dis_pair_list:
                s = '{a}-{b},{c:0.4f}'.format(
                        a=str(cls_id), b=str(label), c=dis)
                line.append(s)
            line = '\t'.join(line)
            f.write(line+'\n')


def write_clusters(path: TextIO, q: DistanceQ):
    """Write down the clusters.
    """
    ans = [-1] * len(q.fix_embeddings)
    ans = np.array(ans)
    indices = torch.nonzero(q.active).reshape(-1)
    indices = indices.cpu().numpy().tolist()
    for i, idx in enumerate(indices):
        t = q.clusters[idx]
        ans[t.indices] = i

    with open(path, 'w', encoding='utf8') as f:
        for i in ans:
            f.write(str(i)+'\n')


def load_clusters(path: TextIO) -> List[List[int]]:
    """ Load the clusters from the file.

    Return:
        reval[i] is the list of points that belong
        to cluster i.
    """
    cluster_labels = []
    with open(path, encoding='utf8') as f:
        for line in f:
            cluster_labels.append(int(line.strip()))
    cluster_num = max(cluster_labels)+1
    reval = [[] for _ in range(cluster_num)]

    for i, v in enumerate(cluster_labels):
        reval[v].append(i)
    return reval


def assign_labels(
        clusters_indices: List[List[int]],
        annotation: np.array) -> List[List[int]]:
    """ Assign labele to each cluster.
    """
    labels = []
    for cls in clusters_indices:
        labs = [annotation[i] for i in cls]
        labels.append(labs)
    return labels


def map_to_label(
        idx2label: dict,
        cluster_list: List[List[Tuple[int, int, float]]],
        real_labels: np.array
        ) -> Tuple[List[List[Tuple[int, str, float]]], List[str]]:
    """Map the int label to str label.

    Args:
        idx2label: A dictionary from int to str.
        cluster_list: cluster_list[i][j] is a tuple of
                      (cluster_id, label, dis),
                      represents the distance between test point i and the
                      cluster with label.
        real_labels: np.array. The real int labels for each test point.
    """
    assert len(cluster_list) == len(real_labels)
    real_labels = [idx2label[v] for v in real_labels]
    return_list = []
    for i, label_dis_pair_list in enumerate(cluster_list):
        s = [(cls_id, idx2label[label], dis)
             for cls_id, label, dis in label_dis_pair_list]
        return_list.append(s)
    return return_list, real_labels


def write_convex_dis(
        path: str,
        label_pairs: List[Tuple[str, str]],
        diss: List[float]):
    """Write distances between clusters into the file.
    """
    assert len(label_pairs) == len(diss)
    with open(path, 'w', encoding='utf8') as f:
        for (cls_i, label_i, cls_j, label_j), dis in zip(label_pairs, diss):
            s = '({a}-{b}, {c}-{d}): {e:0.4f}\n'.format(
                    a=str(cls_i), b=str(label_i),
                    c=str(cls_j), d=str(label_j),
                    e=dis)
            f.write(s)


def write_dis_inside_convex(
        path: TextIO,
        mean_std: List[Tuple[float, float]],
        labels: List[str]):
    assert len(mean_std) == len(labels)
    with open(path, 'w') as f:
        for i in range(len(labels)):
            tag = labels[i]
            mean, std = mean_std[i]
            s = '{a} {b:.4f} {c:.4f}\n'
            s = s.format(a=str(tag), b=mean, c=std)
            f.write(s)


def load_indices(path: TextIO) -> List[List[int]]:
    indices = []
    with open(path) as f:
        for line in f:
            s = [int(t) for t in line.strip().split()]
            indices.append(s)
    return indices


def write_data_cartography(
        path: TextIO,
        data: List[Tuple[float, float, float]]):
    """
    Args:
        data: a list of (mean_prob, std_prob, correctness)
    """
    with open(path, 'w') as f:
        for mean, std, corr in data:
            s = '{a:.3f}\t{b:.3f}\t{c:.3f}\n'.format(
                    a=mean, b=std, c=corr)
            f.write(s)


def write_batch_label_dis_vecs(
        path: TextIO,
        data: np.array,
        idx2label: Dict[int, str]):
    n = len(idx2label)
    indices = [(i, j) for i in range(n) for j in range(i+1, n)]
    assert len(indices) == len(data)
    assert 12 == data.shape[1]
    with open(path, 'w') as f:
        s = [str(i) for i in range(1, 13)]
        s = ','.join(s)
        s = ' ,' + s
        f.write(s+'\n')
        for (i, j), vec in zip(indices, data):
            A = idx2label[i]
            B = idx2label[j]
            s = '{a}---{b},'.format(a=A, b=B)
            v = [format(i, '0.4f') for i in vec]
            s = s + ','.join(v)
            f.write(s+'\n')


def write_batch_diss(
        path: TextIO,
        data: np.array,
        idx2label: Dict[int, str]):
    n = len(data)
    assert n == len(idx2label)
    assert 12 == data.shape[1]
    with open(path, 'w') as f:
        s = [str(i) for i in range(1, 13)]
        s = ','.join(s)
        s = ' ,' + s
        f.write(s+'\n')
        for i, vec in zip(list(range(n)), data):
            A = idx2label[i]
            s = '{a},'.format(a=A)
            v = [format(i, '0.4f') for i in vec]
            s = s + ','.join(v)
            f.write(s+'\n')


def write_diss(
        path: TextIO,
        data: List[float],
        idx2label: Dict[int, str]):
    n = len(data)
    assert n == len(idx2label)
    with open(path, 'w') as f:
        for i, v in zip(list(range(n)), data):
            A = idx2label[i]
            s = '{a},{b:0.4f}'.format(a=A, b=v)
            f.write(s+'\n')


def write_label_dis_vecs(
        path: TextIO,
        data: List[float],
        idx2label: Dict[int, str]
        ):
    n = len(idx2label)
    indices = [(i, j) for i in range(n) for j in range(i+1, n)]
    assert len(indices) == len(data)
    with open(path, 'w') as f:
        for (i, j), v in zip(indices, data):
            A = idx2label[i]
            B = idx2label[j]
            s = '{a}---{b},{c:0.4f}'.format(a=A, b=B, c=v)
            f.write(s+'\n')
