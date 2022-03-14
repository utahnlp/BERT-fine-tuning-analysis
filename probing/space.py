# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Yichu Zhou - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.6.0
#
# Date: 2020-03-20 10:42:42
# Last modified: 2020-12-29 14:30:21

import logging
from typing import List, Tuple

from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
from joblib import Parallel, delayed
# from scipy.spatial import distance
import gurobipy as gp
from gurobipy import GRB
# from sklearn.svm import LinearSVC
from sklearn.svm import SVC

from probing.distanceQ import DistanceQ
from probing.clusters import Cluster

Tensor = torch.Tensor

logger = logging.getLogger(__name__)


class Space:
    def __init__(self, args):
        self._args = args
        self.whitelist = []
        self.blacklist = []
        self.solver = self.check_overlap_detector()

    def check_overlap_detector(self):
        """
        If gurobi is valid, we will use gurobi to solve a LP problem.
        Otherwise, a hard svm will be used to find the hyperplane.
        """
        s1 = ('Gurobi is NOT found in the system, '
              'we will use sklearn.SCV instead.')
        s2 = ('Gurobi IS found in the system, we will use Gurobi.')
        try:
            gp.Model("lp")
            logger.info(s2)
            return Space.lp
        except Exception:
            logger.info(s1)
            return Space.hardSVM

    @staticmethod
    def point2hull(X: np.array, p: np.array) -> float:
        """Compute the distance between cluster X and point p.
        """
        # clf = LinearSVC(tol=1e-5, loss='hinge', C=1000, max_iter=20000)
        clf = SVC(tol=1e-5, C=10000, kernel='linear', max_iter=20000)
        y = [0] * len(X)
        y.append(1)
        y = np.array(y)
        XX = np.concatenate((X, p.reshape(1, -1)))
        clf.fit(XX, y)
        # Can not separate them
        # This means the test point
        # is inside of the convex hull.
        # As a result, the distance is zero.
        if clf.score(XX, y) != 1.0:
            return 0

        # Compute the distance from point to hyperplane
        w = clf.coef_.reshape(-1)
        b = clf.intercept_[0]
        d = np.dot(w, p) + b
        d = abs(d) / np.linalg.norm(w)

        return 2*d

    @staticmethod
    def hull2hull(X1: np.array, X2: np.array) -> float:
        """Compute the distance between two convex hulls.
        """
        clf = SVC(tol=1e-5, C=1000000, kernel='linear', max_iter=50000)
        # clf = LinearSVC(tol=1e-5, loss='hinge', C=100000, max_iter=20000)
        y1 = [1] * len(X1)
        y2 = [-1] * len(X2)
        y1 = np.array(y1)
        y2 = np.array(y2)
        XX = np.concatenate((X1, X2))
        yy = np.concatenate((y1, y2))
        clf.fit(XX, yy)

        score = clf.score(XX, yy)
        # Two convex hulls overlap
        if score != 1:
            return 0

        w = clf.coef_.reshape(-1)
        b = clf.intercept_[0]

        d = np.dot(XX, w) + b
        d = abs(d) / np.linalg.norm(w)

        d = d*2
        return np.min(d)

    def overlapping(self, q: DistanceQ, t: Cluster) -> bool:
        """ Check if the newcluster overlaps with other clusters
            with different labels.

        Return:
            True: there is at least one overlapping.
            False: there is no overlapping.
        """
        # Filter the clusters that have the potential to overlap
        # with t.
        indexs = self._sphere_overlapping(q, t)
        indexs = indexs.cpu().numpy()
        if len(indexs) == 0:
            return False

        # Filter through the black list
        if [i for i in indexs if self._black_cached(q.clusters[i], t)]:
            return True

        # Filter through the white list
        indexs = [i for i in indexs
                  if not self._white_cached(q.clusters[i], t)]

        if len(indexs) == 0:
            return False
        else:
            # Apply the LP solver.
            return self._lp_overlapping(q, t, indexs)

    def _sphere(self, q: DistanceQ, cluster: Cluster) -> Tensor:
        """ Find the clusters that have the potential to overlap
        with the new cluster.
        """
        args = self._args
        # Compute the center for new cluster
        indexs = torch.LongTensor(cluster.indices)
        vecs = q.fix_embeddings[indexs].to(args.device)
        center = torch.mean(vecs, 0)
        center = center.reshape(1, -1)

        # Find the maximum distance and
        # use it as the radius
        dis = torch.cdist(vecs, center)
        r = torch.max(dis).reshape(1)

        embeddings = q.embeddings
        dist = torch.cdist(embeddings, center).reshape(-1)
        assert len(embeddings) == len(dist)

        overlap_mask = (q.radius + r) > dist
        return overlap_mask

    def _sphere_overlapping(self, q: DistanceQ, newcluster: Cluster) -> Tensor:
        """Check if the newcluster overlaps with other clusters
        with different labels in sphere.
        """
        args = self._args
        # Compute the new center and radius
        overlap_mask = self._sphere(q, newcluster)

        # We do not compare the clusters with the same labels
        indexs = torch.ones(q.embeddings.shape[0]).bool()
        idx = q.major_labels == newcluster.major_label
        indexs[idx] = False
        indexs = indexs.to(args.device)
        assert indexs.shape == overlap_mask.shape

        # 1. potential overlap
        # 2. Have different labels
        # 3. It is a active cluster
        mask = overlap_mask & indexs & q.active
        return torch.nonzero(mask).reshape(-1)

    def _lp_overlapping(
            self,
            q: DistanceQ,
            newcluster: Cluster,
            indexs: List[int]) -> bool:
        """Apply the LP solver to check overlapping.

        Return:
            True: there is at least overlapping.
            False: there is not overlapping.
        """
        # logger = logging.getLogger('Probing')
        cur_vecs = q.fix_embeddings[torch.LongTensor(newcluster.indices)]
        X1 = cur_vecs.numpy()
        data = []
        for i in indexs:
            idxs = q.clusters[i].indices
            vecs = q.fix_embeddings[torch.LongTensor(idxs)]
            X2 = vecs.numpy()
            data.append((X1, X2))

        # logger.info('Solving {a} LP...'.format(a=str(len(data))))
        results = Parallel(n_jobs=30, prefer='processes', verbose=0,
                           batch_size='auto')(
            delayed(self.solver)(X1, X2) for X1, X2 in data)

        # Add white list to avoid further computation
        for i, v in enumerate(results):
            cache = self._add_list(newcluster, q.clusters[indexs[i]])
            if v == 0:
                self.whitelist.append(cache)
            else:
                self.blacklist.append(cache)

        if np.sum(results) != 0:
            return True
        else:
            return False

    @staticmethod
    def lp(X1: np.array, X2: np.array) -> int:
        """Return 1 when the LP problem is infeasible.
        """
        # logger = logging.getLogger('Probing')
        m = X1.shape[1]
        model = gp.Model("lp")

        # Create variables
        W = model.addMVar(shape=m+1, lb=-GRB.INFINITY,
                          ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="W")
        # Adding the bias
        XX1 = np.concatenate((X1, np.ones((X1.shape[0], 1))), axis=1)
        XX2 = np.concatenate((X2, np.ones((X2.shape[0], 1))), axis=1)

        Y1 = np.array([1]*X1.shape[0])
        Y2 = np.array([-1]*X2.shape[0])

        model.addConstr(XX1 @ W >= Y1)
        model.addConstr(XX2 @ W <= Y2)
        model.setObjective(0, GRB.MINIMIZE)
        model.setParam('OutputFlag', False)
        model.setParam('FeasibilityTol', 1e-4)

        # Optimize model
        model.update()
        model.optimize()

        # s = 'Solved a LP problem, status code {a}'
        # logger.info(s.format(a=str(model.Status)))
        return int(model.Status != GRB.OPTIMAL)

    def hardSVM(X1: np.array, X2: np.array):
        """Return 1 when the X1 and X2 are not separable.
        """
        clf = SVC(tol=1e-5, C=10000, kernel='linear', max_iter=500000)
        # clf = LinearSVC(tol=1e-5, loss='hinge', C=100000, max_iter=20000)
        y1 = [0] * len(X1)
        y2 = [1] * len(X2)
        y1 = np.array(y1)
        y2 = np.array(y2)
        XX = np.concatenate((X1, X2))
        yy = np.concatenate((y1, y2))

        scaler = StandardScaler()
        scaler.fit(XX)
        XX = scaler.transform(XX)
        clf.fit(XX, yy)

        score = clf.score(XX, yy)
        # Two convex hulls overlap
        if score != 1:
            return 1
        else:
            return 0

    def _add_list(self, A: Cluster, B: Cluster) -> Tuple[set, set]:
        """ Add A and B, along with all their children
        into white list.
        """
        setA = set(A.indices)
        setB = set(B.indices)
        return (setA, setB)

    def _white_cached(self, A: Cluster, B: Cluster) -> bool:
        """Test if cluster A and B have been tested.

        Return true if A and B have been tested.
        """
        a = set(A.indices)
        b = set(B.indices)
        for setA, setB in self.whitelist:
            if (a <= setA and b <= setB) or (b <= setA and a <= setB):
                return True
        else:
            return False

    def _black_cached(self, A: Cluster, B: Cluster) -> bool:
        """Test if any subclusters of A and B have been tested.

        Return true if subclusters of A and B have been tested.
        """
        a = set(A.indices)
        b = set(B.indices)
        for setA, setB in self.blacklist:
            if (setA <= a and setB <= b) or (setA <= b and setB <= a):
                return True
        else:
            return False
