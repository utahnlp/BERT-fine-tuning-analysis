# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Yichu Zhou - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.6.0
#
# Date: 2019-08-16 09:58:30
# Last modified: 2020-11-02 11:59:27

"""
Test for SVM margin.
"""

import numpy as np
# from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import gurobipy as gp
from gurobipy import GRB


def load_embeddings(path):
    reval = []
    with open(path, encoding='utf8') as f:
        for line in f:
            s = line.strip().split()
            vec = [float(v) for v in s]
            reval.append(vec)
    return reval


def distance(X, p):
    # clf = LinearSVC(tol=1e-5, loss='hinge', C=100000, max_iter=20000)
    clf = SVC(tol=1e-5, C=1000000, kernel='linear', max_iter=20000)
    y = [0] * len(X)
    y.append(1)
    y = np.array(y)
    XX = np.concatenate((X, p.reshape(1, -1)))
    clf.fit(XX, y)
    w = clf.coef_.reshape(-1)
    b = clf.intercept_[0]
    if clf.score(XX, y) != 1.0:
        print(w)
        print(b)
        return 0

    print(w)
    print(b)

    d = np.dot(w, p) + b
    d = abs(d) / np.linalg.norm(w)

    d = (d*2)
    return d


def hull2hull(X1, X2):
    clf = SVC(tol=1e-4, C=10000, kernel='linear', max_iter=20000)
    # clf = LinearSVC(tol=1e-5, loss='hinge', C=100000, max_iter=20000)
    y1 = [1] * len(X1)
    y2 = [-1] * len(X2)
    y1 = np.array(y1)
    y2 = np.array(y2)
    XX = np.concatenate((X1, X2))
    yy = np.concatenate((y1, y2))
    clf.fit(XX, yy)

    w = clf.coef_.reshape(-1)
    b = clf.intercept_[0]
    print(w)
    print(b)
    score = clf.score(XX, yy)
    if score != 1:
        return 0

    d = np.dot(XX, w) + b
    d = abs(d) / np.linalg.norm(w)

    d = (d*2)
    return np.min(d)


def lp(X1, X2, has_same=False):
    """Return 1 when the LP problem is infeasible.
    """
    if has_same:
        dist = distance.cdist(X1, X2)
        # There is a same vector on both sides.
        if np.any(dist == 0):
            return 1
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
    # model.setParam('Method', 2)
    # model.setParam('FeasibilityTol', 1e-4)

    # Optimize model
    model.update()
    model.optimize()

    print(model.Status)
    return int(model.Status != GRB.OPTIMAL)


if __name__ == '__main__':
    # X = [[1, 1], [2, 2], [3, 3], [4, 4],
    #      [1, 2], [3, 2]]
    # X = [[5, 10], [0, 5], [10, 5], [5, 0]]
    # p = [10, -5]
    # X = np.array(X)
    # p = np.array(p)

    # d = distance(X, p)
    # print(d)

    X1 = [[1, 6], [1, 3], [15, 3], [15, 6]]
    X2 = [[1, -3], [1, -1], [2, -3], [2, -1]]
    X1 = np.array(X1)
    X2 = np.array(X2)
    d = hull2hull(X1, X2)
    print(d)
