# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Yichu Zhou - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.6.0
#
# Date: 2020-04-01 10:23:10
# Last modified: 2020-12-29 11:32:13

"""
Distance Q implementation.

This is the main class to maintain the list of clusters.
"""
import logging
from typing import List
import heapq

import torch
from tqdm import tqdm
from tqdm import trange
import numpy as np

from probing.clusters import ClusterDisList
from probing.clusters import ClusterDisPair
from probing.clusters import Cluster
from probing.config import Config

logger = logging.getLogger(__name__)


class DistanceQ:
    """ This distance data structure has the
        following functions:
    1. Return the min.
    2. Delete Clusters
    3. Add new Clusters.
    """
    def __init__(
            self,
            config: Config,
            embeddings: np.array,
            init_clusters: List[Cluster],
            label_size: int):
        """In this class,  there are a few attributes need to be
        maintained:

            - clusters: a list of clusters
            - radius: record the radius for each cluster
            - embeddings: record the center for each cluster
            - active: record which cluster is active
            - major_labels: record the majority_label for each cluster
        """
        # logger = logging.getLogger('Probing')
        # logger.info('Initialize the Distance Queue...')
        self._args = config
        args = config
        self.label_size = label_size
        self.clusters = list(init_clusters)

        # self.fix_embeddings is the original embedding
        # and it should not be changed during the process
        self.fix_embeddings = torch.Tensor(embeddings)

        # Initialize the centers for each clusters
        radius = []
        centers = []
        for cls in self.clusters:
            indexs = cls.indices
            vecs = self.fix_embeddings[indexs].to(args.device)
            center = torch.mean(vecs, 0)
            center = center.reshape(1, -1)
            centers.append(center)
            # Find the maximum distance and
            # use it as the radius
            dis = torch.cdist(vecs, center)
            r = torch.max(dis).reshape(1)
            radius.append(r)
        self.radius = torch.cat(radius).to(args.device)

        # self.embeddings is the centers of each cluster
        # It dynamic changes during the merging process
        self.embeddings = torch.cat(centers).to(args.device)
        assert self.radius.shape[0] == len(self.clusters)
        assert self.embeddings.shape[0] == len(self.clusters)

        # Initalize the major labels
        major_labels = [t.major_label for t in init_clusters]
        self.major_labels = torch.IntTensor(major_labels).to(config.device)

        self.active = torch.ones(
                len(self.clusters)).bool().to(config.device)
        self.heap_built = False

    def remove_pair(self, i, j) -> None:
        """ Clusrer i and cluster j is merged and
        they should be removed from the lists.
        """
        self.remove(i)
        self.remove(j)
        if self.heap_built:
            heapq.heapify(self.cluster_dis)

    def remove(self, idx: int) -> None:
        """ Remove all the records regrading cluster idx.
        """
        self.active[idx] = False
        if self.heap_built:
            self.dis_maps[idx].deactive()

    def min(self) -> ClusterDisPair:
        """ Return and delete the pair with least distance.
        """
        while True:
            # Find the list of pairs that has the minimum distance.
            dislist = heapq.heappop(self.cluster_dis)
            # Return and remove the minimum cluster pair in this list.
            pair = dislist.min()
            # Push back the list
            heapq.heappush(self.cluster_dis, dislist)
            # Check if the given pair is valid
            if self.active[pair.i] and self.active[pair.j]:
                return pair

    def add(self, newcluster):
        """Add a new cluster.
        """
        args = self._args
        self.clusters.append(newcluster)

        # Compute the new center
        indexs = torch.LongTensor(newcluster.indices)
        vecs = self.fix_embeddings[indexs].to(args.device)
        center = torch.mean(vecs, 0)
        center = center.reshape(1, -1)

        # add cluster distance pairs
        if self.heap_built:
            diss = torch.cdist(self.embeddings[self.active], center)
            diss = diss.reshape(-1).cpu().numpy().tolist()
            nonzeros = torch.nonzero(self.active).reshape(-1)
            nonzeros = nonzeros.cpu().numpy().tolist()
            tmp = []
            m = len(self.clusters)-1

            # Build the new distance list for the new cluster:
            # computing all pair-wise distance for other clusers
            # that have the same label.
            for i, v in enumerate(nonzeros):
                if self.clusters[v].major_label == newcluster.major_label:
                    tmp.append(ClusterDisPair(v, m, diss[i]))
            tmp = ClusterDisList(tmp, m)
            self.dis_maps[m] = tmp
            heapq.heappush(self.cluster_dis, tmp)

        # Update the active
        active = torch.Tensor([True]).bool().to(args.device)
        self.active = torch.cat((self.active, active))

        # Update the centers
        self.embeddings = torch.cat((self.embeddings, center), 0)

        dis = torch.cdist(vecs, center)
        r = torch.max(dis).reshape(1)
        self.radius = torch.cat((self.radius, r))

        # Update the major labels
        m = torch.IntTensor([newcluster.major_label]).to(args.device)
        self.major_labels = torch.cat((self.major_labels, m))

        assert len(self.embeddings) == len(self.active)
        assert len(self.clusters) == len(self.active)
        assert len(self.radius) == len(self.active)
        assert len(self.major_labels) == len(self.active)
        if self.heap_built:
            assert len(self.cluster_dis) == len(self.active)

    def __len__(self):
        return len(torch.nonzero(self.active))

    def build_heaps(self):
        if self.heap_built:
            return
        args = self._args
        logger.info('Initializing the pair-wise distance...')

        logger.info('Categorize the clusters based on the label...')
        labels_indices = [[] for _ in range(self.label_size)]
        for i, t in enumerate(self.clusters):
            labels_indices[t.major_label].append(i)

        logger.info(
                'Computing the pair-wise distance inside the same label...')
        tmp = [[] for _ in range(len(self.clusters))]
        for i in trange(len(labels_indices), desc='Labels'):
            indices = labels_indices[i]
            idx = torch.LongTensor(indices).to(args.device)
            embeds = self.embeddings[idx]
            # Compute the pair-wise distance between
            # all clusters that have the same label.
            pdist = torch.nn.functional.pdist(embeds)
            pdist = pdist.cpu().numpy().tolist()
            m = len(embeds)
            ij = torch.triu_indices(m, m, 1)
            ij = ij.cpu().numpy().tolist()

            s = 'Building list for label {a}'.format(a=str(i))
            for k in trange(len(ij[0]), desc=s, leave=False):
                i = min(indices[ij[0][k]], indices[ij[1][k]])
                j = max(indices[ij[0][k]], indices[ij[1][k]])
                t = ClusterDisPair(i, j, pdist[k])
                tmp[j].append(t)

        self.dis_maps = dict()
        cluster_dis = []
        assert len(tmp[0]) == 0

        logger.info('Build the double heaps...')
        for i in tqdm(range(len(tmp))):
            t = ClusterDisList(tmp[i], i)
            self.dis_maps[i] = t
            heapq.heappush(cluster_dis, t)
        self.cluster_dis = cluster_dis
        self.heap_built = True

    @staticmethod
    def cleanQ(args, q: 'DistanceQ'):
        """ Build a new clean Q.
        """
        indices = torch.nonzero(q.active).reshape(-1)
        clusters = [q.clusters[i] for i in indices]
        return DistanceQ(args, q.fix_embeddings,
                         clusters, q.label_size)
