# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Yichu Zhou - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.8.0
#
# Date: 2021-02-05 09:18:42
# Last modified: 2021-03-03 09:32:32

"""
Batch probing for a series embedding spaces.
"""
import logging
from pathlib import Path

# import ExAssist as EA
import numpy as np

from probing import utils
from probing import loader
from probing.analyzer import Analyzer
from probing.probing import Probe
from probing.clusters import Cluster
from probing.distanceQ import DistanceQ

logger = logging.getLogger(__name__)


def batch_analyze(config):
    batch_path = Path(config.batch_embeddings_path)
    common_label_vec_path = Path(config.batch_label_vec_path)
    common_inside_mean_path = Path(config.batch_inside_mean_path)
    common_inside_max_path = Path(config.batch_inside_max_path)
    common_outside_min_path = Path(config.batch_outside_min_path)

    if not common_label_vec_path.exists():
        common_label_vec_path.mkdir()
    if not common_inside_mean_path.exists():
        common_inside_mean_path.mkdir()
    if not common_inside_max_path.exists():
        common_inside_max_path.mkdir()
    if not common_outside_min_path.exists():
        common_outside_min_path.mkdir()

    iterations = list(batch_path.iterdir())

    for iter_path in iterations:
        label_vec_path = common_label_vec_path / (iter_path.name + '.txt')
        inside_mean_path = common_inside_mean_path / (iter_path.name + '.txt')
        outside_min_path = common_outside_min_path / (iter_path.name + '.txt')
        inside_max_path = common_inside_max_path / (iter_path.name + '.txt')

        label_dis_vecs = []
        inside_mean_diss = []
        outside_min_diss = []
        inside_max_diss = []
        for layer_idx in range(1, 13):
            layer_path = iter_path / (str(layer_idx) + '.txt')
            config.embeddings_path = layer_path
            logger.info('Loading embeddings...')
            data = loader.load_train(config)
            annotations = data[0]
            labels = data[1]
            embeddings = data[2]
            label2idx = data[3]

            logger.info('Loading clusters...')
            clusters_indices = utils.load_clusters(config.probing_cluster_path)
            labels_list = utils.assign_labels(clusters_indices, annotations)
            assert len(clusters_indices) == len(labels_list)
            # Filter the clusters with points less than 5
            clusters = [Cluster(indices, labs) for
                        indices, labs in zip(clusters_indices, labels_list)
                        if len(indices) > 5]
            q = DistanceQ(config, embeddings, clusters, len(labels))
            logger.info('Finish loading the clusters...')

            probe = Probe(config)
            # If overlaps happen
            # if True:
            if not probe._check_overlaps(q):
                s = str(layer_path) + ' IS linearly separable'
                logger.info(s)
                analyzer = Analyzer(config)
                idx2label = {value: key for key, value in label2idx.items()}

                logger.info('Computing the distances between clusters...')
                total_label_pair_dis = analyzer.convex2convex(q)
                logger.info('Computing the distances vectors...')
                label_dis_vec = analyzer.label_dis_vec(total_label_pair_dis)
                logger.info('Computing the outside min distance...')
                outside_min_dis = analyzer.outside_min_dis(
                        total_label_pair_dis)
                logger.info('Computing the inside mean distance...')
                inside_mean_dis = analyzer.inside_mean_dis(q)
                logger.info('Computing the inside max distance...')
                inside_max_dis = analyzer.inside_max_dis(q)
            else:
                s = str(layer_path) + ' is NOT linearly separable'
                logger.info(s)
                label_dis_vec = [0] * ((q.label_size-1)*q.label_size // 2)
                outside_min_dis = [0] * q.label_size
                inside_mean_dis = [0] * q.label_size
                inside_max_dis = [0] * q.label_size

            if len(label_dis_vecs) != 0:
                assert len(label_dis_vecs[-1]) == len(label_dis_vec)
            if len(outside_min_diss) != 0:
                assert len(outside_min_diss[-1]) == len(outside_min_dis)
            if len(inside_mean_diss) != 0:
                assert len(inside_mean_diss[-1]) == len(inside_mean_dis)
            if len(inside_max_diss) != 0:
                assert len(inside_max_diss[-1]) == len(inside_max_dis)

            label_dis_vecs.append(label_dis_vec)
            outside_min_diss.append(outside_min_dis)
            inside_mean_diss.append(inside_mean_dis)
            inside_max_diss.append(inside_max_dis)

        label_dis_vecs = np.array(label_dis_vecs).transpose()
        outside_min_diss = np.array(outside_min_diss).transpose()
        inside_mean_diss = np.array(inside_mean_diss).transpose()
        inside_max_diss = np.array(inside_max_diss).transpose()

        # print(label_dis_vecs.shape)
        # print(outside_min_diss.shape)
        # print(inside_mean_diss.shape)
        utils.write_batch_label_dis_vecs(
                label_vec_path, label_dis_vecs, idx2label)
        utils.write_batch_diss(
                outside_min_path, outside_min_diss, idx2label)
        utils.write_batch_diss(
                inside_mean_path, inside_mean_diss, idx2label)
        utils.write_batch_diss(
                inside_max_path, inside_max_diss, idx2label)
