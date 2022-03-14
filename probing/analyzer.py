# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Yichu Zhou - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.8.0
#
# Date: 2020-12-29 13:46:09
# Last modified: 2021-03-03 10:48:44

"""
Analyzing functions.
"""

import logging
from typing import List, Tuple
import collections

from tqdm import tqdm
from joblib import Parallel, delayed
import torch

import numpy as np
from probing.distanceQ import DistanceQ
from probing.space import Space

logger = logging.getLogger(__name__)


class Analyzer:
    def __init__(self, config):
        self.args = config

    def predict(self, q, ann, embeds):
        return self.points2convex(q, ann, embeds)

    def points2convex(
            self,
            q: DistanceQ,
            ann: np.array,
            embeds: np.array
            ) -> Tuple[float, List[List[Tuple[int, int, float]]]]:
        """
        Make predictions for `embeds` based on the distances
        between each point in `embeds` and all the clusters.

        Returns:
            - List((cluster_id, major_label, distance)):
                    the ranking of clusters for each test point
                    based on the distance.
        """
        assert len(ann) == len(embeds)
        clusters = q.clusters

        logger.info('Computing the distances...')
        return_list = []
        correct = 0
        for i, (label, vec) in tqdm(
                enumerate(zip(ann, embeds)), total=len(ann)):
            data = []
            diss = []
            # select all the points belong to cluster j
            for j in range(len(clusters)):
                cls = clusters[j]
                indexs = torch.LongTensor(cls.indices)
                vecs = q.fix_embeddings[indexs]
                vecs = vecs.cpu().numpy()
                data.append((vecs, vec))
            diss = Parallel(n_jobs=20, prefer='processes', verbose=0,
                            batch_size='auto')(
                delayed(Space.point2hull)(X1, X2) for X1, X2 in data)

            diss = np.array(diss)
            sorted_indices = np.argsort(diss)
            preds = [
                    (j, clusters[j].major_label, diss[j])
                    for j in sorted_indices]

            if preds[0][1] == label:
                correct += 1
            return_list.append(preds)
        acc = correct / len(ann)
        return acc, return_list

    def convex2convex(
            self,
            q: DistanceQ
            ) -> np.array:
        """Return the distance between the clusters.

        total_label_pair_dis[i][j] means the distance
        between label i and labe j.
        """
        data = []
        clusters = q.clusters

        indices = list(range(len(clusters)))

        # Prepare the embeddings
        for i in range(len(q.clusters)):
            cls = q.clusters[i]
            indexs = torch.LongTensor(cls.indices)
            vecs = q.fix_embeddings[indexs]
            vecs = vecs.cpu().numpy()
            data.append(vecs)

        total_label_pair_dis = np.empty((q.label_size, q.label_size))
        total_label_pair_dis.fill(np.inf)
        indexs = [(i, j) for i in indices for j in indices
                  if i < j]
        # Only compute the distances between clusters with different labels
        indexs = [(i, j) for i, j in indexs
                  if clusters[i].major_label != clusters[j].major_label]

        data = [(data[i], data[j]) for i, j in indexs]
        label_pairs = [(clusters[i].major_label, clusters[j].major_label)
                       for i, j in indexs]

        diss = Parallel(n_jobs=10, prefer='processes', verbose=0,
                        batch_size=1)(
            delayed(Space.hull2hull)(X1, X2) for X1, X2 in data)

        assert len(diss) == len(label_pairs)
        label_holder = collections.defaultdict(list)
        for (i, j), d in zip(label_pairs, diss):
            key = (min(i, j), max(i, j))
            label_holder[key].append(d)

        for (i, j), ds in label_holder.items():
            d = np.mean(ds)
            total_label_pair_dis[i][j] = d
            total_label_pair_dis[j][i] = d
        return total_label_pair_dis

    def label_dis_vec(
            self,
            total_label_pair_dis: np.array) -> List[float]:
        reval = []
        n = total_label_pair_dis.shape[0]
        for i in range(n):
            for j in range(i+1, n):
                if total_label_pair_dis[i][j] != np.inf:
                    reval.append(total_label_pair_dis[i][j])
                else:
                    reval.append(0)
        return reval

    def inside_max_dis(
            self,
            q: DistanceQ) -> List[float]:
        """Computing the max distance inside each cluster.
        """
        data = []
        labels = []
        for i in range(len(q.clusters)):
            cls = q.clusters[i]
            indexs = torch.LongTensor(cls.indices)
            vecs = q.fix_embeddings[indexs].to(self.args.device)
            data.append(vecs)
            labels.append(cls.major_label)

        holder = collections.defaultdict(list)
        for tag, embeds in zip(labels, data):
            pdist = torch.nn.functional.pdist(embeds)
            if len(pdist) > 0:
                holder[tag].append(torch.max(pdist).cpu().numpy())

        reval = [0] * q.label_size
        for tag, ds in holder.items():
            reval[tag] = np.max(ds)
        return reval

    def inside_mean_dis(
            self,
            q: DistanceQ) -> List[float]:
        """Computing the mean distance inside each cluster.
        """
        data = []
        labels = []
        for i in range(len(q.clusters)):
            cls = q.clusters[i]
            indexs = torch.LongTensor(cls.indices)
            vecs = q.fix_embeddings[indexs].to(self.args.device)
            data.append(vecs)
            labels.append(cls.major_label)

        holder = collections.defaultdict(list)
        for tag, embeds in zip(labels, data):
            pdist = torch.nn.functional.pdist(embeds)
            if len(pdist) > 0:
                holder[tag].append(torch.mean(pdist).cpu().numpy())

        reval = [0] * q.label_size
        for tag, ds in holder.items():
            reval[tag] = np.mean(ds)
        return reval

    def outside_min_dis(
            self,
            total_label_pair_dis: np.array) -> List[float]:
        """Find the minmum distance for each label.
        """
        min_dis = []
        for t in total_label_pair_dis:
            min_dis.append(np.min(t))
            if min_dis[-1] == np.inf:
                min_dis[-1] = 0
        assert len(min_dis) == total_label_pair_dis.shape[0]
        return min_dis
