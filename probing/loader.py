# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Yichu Zhou - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.8.0
#
# Date: 2021-01-06 10:07:08
# Last modified: 2021-02-05 10:10:56

"""
Load data
"""

from typing import Tuple
import logging

import numpy as np

from probing import utils
from probing.config import Config

logger = logging.getLogger(__name__)


def load_train(
        config: Config) -> Tuple[np.array, np.array, np.array]:
    """Loading all the necessary input files.

    This function load 3 files:
        - entities: A file contains the entities and labels.
                    One entity per line.
        - label_set_path: A file contains all the possible labels.
                          We have a separate file because in some cases,
                          not all the labels occure in the training set.
        - embeddings_path: A file contains all the embeddings.
                           A vector per line.
    """
    path = config.entities_path
    logger.info('Load entities from ' + path)
    entities = utils.load_entities(path)

    # For debugging
    n = len(entities)
    # n = 200
    annotations = [entities[i].Label for i in range(n)]
    entities = [entities[i] for i in range(n)]

    s = 'Finish loading {a} entities...'
    s = s.format(a=str(len(entities)))
    logger.info(s)

    labels = sorted(list(utils.load_labels(config.label_set_path)))
    label2idx = {labels[i]: i for i in range(len(labels))}
    annotations = [label2idx[t] for t in annotations]

    logger.info('Label size={a}'.format(a=str(len(labels))))

    embeddings_path = config.embeddings_path
    logger.info('Loading embeddings from ' + str(embeddings_path))
    embeddings = utils.load_embeddings(embeddings_path)
    embeddings = embeddings[:n]
    logger.info('Finish loading embeddings...')

    assert len(embeddings) == n

    annotations = np.array(annotations)
    labels = np.array(labels)
    embeddings = np.array(embeddings)
    return annotations, labels, embeddings, label2idx


def load_test(config: Config):
    path = config.test_entities_path
    logger.info('Load entities from ' + path)
    entities = utils.load_entities(path)

    # For debugging
    n = len(entities)
    # n = 30
    annotations = [entities[i].Label for i in range(n)]
    entities = [entities[i] for i in range(n)]

    s = 'Finish loading {a} entities...'
    s = s.format(a=str(len(entities)))
    logger.info(s)

    labels = sorted(list(utils.load_labels(config.label_set_path)))
    label2idx = {labels[i]: i for i in range(len(labels))}
    annotations = [label2idx[t] for t in annotations]

    embeddings_path = config.test_embeddings_path
    logger.info('Loading embeddings from ' + embeddings_path)
    embeddings = utils.load_embeddings(embeddings_path)
    embeddings = embeddings[:n]
    logger.info('Finish loading embeddings...')

    assert len(embeddings) == n

    annotations = np.array(annotations)
    labels = np.array(labels)
    embeddings = np.array(embeddings)
    assert len(annotations) == len(embeddings)
    return annotations, embeddings, label2idx
