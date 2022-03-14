# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Yichu Zhou - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.6.0
#
# Date: 2019-08-14 13:44:20
# Last modified: 2022-03-14 11:18:02

"""
Load configuation from file.
"""

from pathlib import Path

import torch


class Config:
    def __init__(self, config):
        self._get_runpath(config)

        self._get_data(config)

        self._get_clustering(config)

    def _get_clustering(self, config):
        self.mode = config.mode
        self.probing_cluster_path = config.probing_cluster_path
        self.enable_cuda = bool(config.enable_cuda)
        self.rate = float(config.rate)
        # self.iter_step = int(config.iter_step)
        cuda = self.enable_cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if cuda else
                                   'cpu')

    def _get_runpath(self, config):
        output_path = Path(config.output_path)
        if not output_path.exists():
            output_path.mkdir(parents=True)
        else:
            if len(list(output_path.iterdir())) != 0:
                raise Exception('The results directory is non-empty!')
        self.cluster_path = output_path / 'clusters.txt'
        self.log_path = output_path / 'log.txt'
        self.prediction_path = output_path / 'prediction.txt'

        # Data cartography
        self.data_map_predictions_path = output_path / 'data_map_predictions/'
        self.data_map_path = output_path / 'data_map.txt'

        # Batch probing
        self.batch_label_vec_path = output_path / 'batch_vec/'
        self.batch_inside_mean_path = output_path / 'inside_mean/'
        self.batch_inside_max_path = output_path / 'inside_max/'
        self.batch_outside_min_path = output_path / 'outside_min/'

        self.label_vec_path = output_path / 'vec.txt'
        self.inside_mean_path = output_path / 'inside_mean.txt'
        self.inside_max_path = output_path / 'inside_max.txt'
        self.outside_min_path = output_path / 'outside_min.txt'

    def _get_data(self, config):
        self.entities_path = config.entities_path
        # self.test_entities_path = config.test_entities_path
        self.label_set_path = config.label_set_path
        self.embeddings_path = config.embeddings_path

        self.batch_embeddings_path = config.batch_embeddings_path
        # self.test_embeddings_path = config.test_embeddings_path
        # Data cartography
        # self.train_indices_path = config.train_indices_path
        # self.test_indices_path = config.test_indices_path
