# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Yichu Zhou - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.6.0
#
# Date: 2020-02-18 11:05:08
# Last modified: 2021-03-04 11:30:44

"""
Applying the probing process.
"""

import logging

from typing import Tuple
import torch
import numpy as np
from tqdm import tqdm
import ExAssist as EA

from probing.space import Space
from probing.clusters import Cluster
from probing.distanceQ import DistanceQ
from probing.config import Config


logger = logging.getLogger(__name__)


class Probe:
    """ The clustering stratgy:
    1. Keep merging to the end.
    2. We check for overlapping at the end.
    3. If there is overlapping, we trace back to the step
       where first error happens.
    4. If there is not overlapping, directly return the result.

    The reason we are doing this because we want to avoid
    unnecessary overlapping checking as many as possible.
    The overlapping checking is expensive.
    """
    def __init__(self, config: Config):
        self._args = config
        self.space = Space(self._args)

    def probe_to_end(self, q: DistanceQ) -> np.array:
        """ Probing to end without any overlapping checking.
        """
        q.build_heaps()
        max_num_steps = len(q)
        # Initialize the tracks...
        logger.info('Initializing the tracks...')
        tracks = [self._snapshot(q)]

        logger.info('Probing to the end...')
        iterator = tqdm(range(max_num_steps))
        for step in iterator:
            try:
                pair = q.min()
            except IndexError:
                iterator.close()
                break
            i, j = pair.i,  pair.j

            # This is for debuging. It should not return a pair
            # whose major_label is different.
            if q.clusters[i].major_label != q.clusters[j].major_label:
                logger.error('Cluster pairs have different major_label!')
                continue
            newcluster = Cluster.merge(q.clusters[i], q.clusters[j])

            # It is important to remove first and then add
            q.remove_pair(i, j)
            q.add(newcluster)

            # Save every step
            tracks.append(np.copy(tracks[-1]))
            tracks[-1][newcluster.indices] = len(q.clusters)-1
        logger.info('Finish probing to the end...')
        s = tracks[-1].tolist()
        assert len(set(s)) >= len(q)
        return tracks

    def _snapshot(self, q: DistanceQ) -> np.array:
        """ Take a snapshot of current DistanceQ.

        This method return a array `tracks` whose length
        is equal to the length of embedding points.

        tracks[i] represents the cluster index that i-th
        embedding belong to.
        """
        indices = torch.nonzero(q.active).reshape(-1)
        indices = indices.cpu().numpy()
        tracks = list(range(len(q.fix_embeddings)))
        tracks = np.array(tracks)
        for i, idx in enumerate(indices):
            t = q.clusters[idx]
            tracks[t.indices] = idx
        return tracks

    def probing(self, q: DistanceQ) -> DistanceQ:
        """ Apply probing procedure
        """
        assist = EA.getAssist('Probing')
        tracks = self.probe_to_end(q)
        logger.info('Checking for the end state..')
        t = self._check_overlaps(q)
        if not t:
            logger.info('There is no overlaps in the end state!')
            logger.info('This space is linear!')
            assist.result['linear'] = 1
            s = 'Final number of clusters: {a}'
            s = s.format(a=str(len(q)))
            logger.info(s)
            return q

        logger.info('This space is non-linear!')
        assist.result['linear'] = 0
        # Use coarse search to find the range of first error
        i, j = self._find_coarse_range(tracks, q)
        # Use binary search to find the first error
        k = self._find_first_error(q, tracks, i, j)

        s = 'Found {a}-th state is the first error state'
        s = s.format(a=str(k-1))
        logger.info(s)

        # rebuild the DistanceQ using the tracks
        # before the first error.
        q = self._build_q_from_track(q, tracks[k-1])

        # Keep merging and checking.
        q = self._forward(q)

        s = 'Final number of clusters: {a}'
        s = s.format(a=str(len(q)))
        logger.info(s)
        return q

    def _check_overlaps(self, q: DistanceQ) -> bool:
        """ Check pair-wise clusters overlapping.

        Return true if there is at least one overlap.
        """
        indexs = torch.nonzero(q.active).reshape(-1)
        indexs = indexs.cpu().numpy()
        logger.info('Start ovelapping checking...')
        iterator = tqdm(indexs)
        flag = False

        for i in iterator:
            if self.space.overlapping(q, q.clusters[i]):
                flag = True
                # Here we do not use early stop
                # because we want to cache all the errors
                # in space object.
                # It may take time here, but save more latter.
                # return flag
        return flag

    def _build_clusters_from_track(
            self,
            q: DistanceQ,
            track: np.array):
        assert len(track) == len(q.fix_embeddings)
        cls_set = set(track.tolist())
        clusters = []

        # Collect clusters.
        for n in cls_set:
            clusters.append(q.clusters[n])

        # make sure the number of embedding points is correct
        s = [len(t.indices) for t in clusters]
        assert sum(s) == len(track)
        return clusters

    def _find_coarse_range(
            self,
            tracks: np.array,
            q: DistanceQ) -> Tuple[int, int]:
        """Return of range of state with the first
        error state in it.

        Returns: (i, j)
        """
        m = len(tracks)-1
        if m < 1000:
            return 0, m
        step = int(m * self._args.rate)+1  # avoid 0 step
        logger.info('Start coarse search...')
        k = m - step
        while True:
            logger.info('Test for state {a}'.format(a=str(k)))
            newq = self._build_q_from_track(q, tracks[k])
            t = self._check_overlaps(newq)
            if not t:
                s = 'Found {a}-th state is correct...'
                s = s.format(a=str(k))
                logger.info(s)
                return (k, k+step)
            else:
                k = k - step
                k = max(0, k)

    def _forward(self, q: DistanceQ) -> DistanceQ:
        """ Merging with overlapping checking.
        """
        q.build_heaps()
        m = len(q)
        logger.info('Start normal forward probing...')
        iterator = tqdm(range(m))
        for _ in iterator:
            try:
                pair = q.min()
            except IndexError:
                iterator.close()
                return q
            i, j = pair.i,  pair.j
            newcluster = Cluster.merge(q.clusters[i], q.clusters[j])
            t = self.space.overlapping(q, newcluster)
            if not t:
                q.remove_pair(i, j)
                q.add(newcluster)
        return q

    # def _closest_set(self, q, test_vec):
    #     args = self._args
    #     embeds = q.fix_embeddings.to(args.device)
    #     vec = torch.Tensor(test_vec).to(args.device)

    #     cdist = torch.cdist(embeds, vec.reshape(1, -1))
    #     cdist = cdist.reshape(-1).cpu().numpy()
    #     min_dists = []
    #     for t in q.clusters:
    #         min_dists.append(min(cdist[t.indices]))
    #     n = min(len(q.clusters), 5)
    #     return np.argsort(min_dists)[:n]

    def _build_q_from_track(
            self,
            q: DistanceQ,
            track: np.array) -> DistanceQ:
        clusters = self._build_clusters_from_track(q, track)
        newq = DistanceQ(
                self._args, q.fix_embeddings,
                clusters, q.label_size)
        return newq

    def _find_first_error(
            self,
            q: DistanceQ,
            tracks: np.array,
            i: int,
            j: int) -> int:
        """Use binary search to find the first error.
        """
        logger.info('Start fine search...')
        while i < j:
            k = (i+j) // 2
            logger.info('Test for state {a}'.format(a=str(k)))
            newq = self._build_q_from_track(q, tracks[k])
            t = self._check_overlaps(newq)
            if t:
                j = k
            else:
                i = k + 1
        return j
