# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Yichu Zhou - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.6.0
#
# Date: 2019-07-24 10:36:21
# Last modified: 2022-03-14 11:23:34

"""
Main enterance.
"""
import logging
import logging.config
import configparser
import numpy as np

import ExAssist as EA

from probing import utils
from probing.config import Config
from probing.probing import Probe
from probing.clusters import Cluster
from probing.distanceQ import DistanceQ
from probing.analyzer import Analyzer
import probing.logconfig as cfg
from probing import loader
from probing import batch_probing as bp

logger = logging.getLogger(__name__)


def run(config):
    annotations, labels, embeddings, label2idx = loader.load_train(config)
    probe = Probe(config)

    clusters = [Cluster([i], [label]) for
                i, label in enumerate(annotations)]

    logger.info('Initialize the Distance Queue...')
    q = DistanceQ(config, embeddings, clusters, len(labels))
    q = probe.probing(q)
    assist = EA.getAssist('Probing')
    assist.result['final number'] = len(q)
    logger.info('Dumping the clusters...')
    utils.write_clusters(config.cluster_path,  q)
    logger.info('Finish dumping the clusters...')

    config.probing_cluster_path = config.cluster_path
    develop_run(config)


def develop_run(config):
    assist = EA.getAssist('Probing')
    s = 'Loading the clusters from {a}'
    s = s.format(a=str(config.probing_cluster_path))
    logger.info(s)
    annotations, labels, embeddings, label2idx = loader.load_train(config)
    clusters_indices = utils.load_clusters(config.probing_cluster_path)
    labels_list = utils.assign_labels(clusters_indices, annotations)
    assert len(clusters_indices) == len(labels_list)
    clusters = [Cluster(indices, labs) for
                indices, labs in zip(clusters_indices, labels_list)
                if len(indices) > 5]
    q = DistanceQ(config, embeddings, clusters, len(labels))
    logger.info('Finish loading the clusters...')

    analyzer = Analyzer(config)
    # annotations, embeddings, label2idx = loader.load_test(config)
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

    utils.write_label_dis_vecs(
            config.label_vec_path, label_dis_vec, idx2label)
    utils.write_diss(
            config.outside_min_path, outside_min_dis, idx2label)
    utils.write_diss(
            config.inside_mean_path, inside_mean_dis, idx2label)
    utils.write_diss(
            config.inside_max_path, inside_max_dis, idx2label)

    outside_min_dis = np.array(outside_min_dis)
    inside_mean_dis = np.array(inside_mean_dis)
    inside_max_dis = np.array(inside_max_dis)
    min_dis = np.min(outside_min_dis[outside_min_dis > 0])
    mean_inside_mean_dis = np.mean(inside_mean_dis[inside_mean_dis > 0])
    max_inside_max_dis = np.max(inside_max_dis[inside_max_dis > 0])

    s = 'global min dis={a}'.format(a=str(min_dis))
    logger.info(s)
    s = 'mean_inside_mean_dis={a}'.format(a=str(mean_inside_mean_dis))
    logger.info(s)
    s = 'max inside max dis={a}'.format(a=str(max_inside_max_dis))
    logger.info(s)

    max_min_ratio = []
    for i, j in zip(inside_max_dis, outside_min_dis):
        if i != 0 and j != 0:
            max_min_ratio.append(i/j)
    s = 'Inside max / outside min = {a}'.format(a=str(np.mean(max_min_ratio)))
    logger.info(s)

    mean_min_ratio = []
    for i, j in zip(inside_mean_dis, outside_min_dis):
        if i != 0 and j != 0:
            mean_min_ratio.append(i/j)
    s = 'Inside mean / outside min = {a}'.format(
            a=str(np.mean(mean_min_ratio)))
    logger.info(s)

    assist.result['global min dis'] = min_dis
    assist.result['mean inside mean dis'] = mean_inside_mean_dis
    assist.result['max inside max dis'] = max_inside_max_dis
    assist.result['max_min_ratio'] = np.mean(max_min_ratio)
    assist.result['mean_min_ratio'] = np.mean(mean_min_ratio)


def main():
    assist = EA.getAssist('Probing')
    assist.deactivate()

    config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation())
    config.read('./config.ini', encoding='utf8')
    # config = args.get_args()

    assist.set_config(config)
    with EA.start(assist) as assist:
        config = Config(assist.config)
        cfg.set_log_path(config.log_path)
        logging.config.dictConfig(cfg.LOGGING_CONFIG)
        if config.mode == 'prediction':
            develop_run(config)
        elif config.mode == 'probing':
            run(config)
        elif config.mode == 'batch_probing':
            bp.batch_analyze(config)


if __name__ == '__main__':
    # import cProfile
    # cProfile.run('main()', sort='cumulative')
    main()
