# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Yichu Zhou - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.8.0
#
# Date: 2020-12-29 15:59:08
# Last modified: 2020-12-29 16:16:42

"""
Setting the args.
"""
import argparse


def get_args():
    parser = argparse.ArgumentParser(
            description='Probing the given embeddings')
    # Run section
    parser.add_argument('-run_comments', '--run_comments')
    parser.add_argument('-run_output_path', '--run_output_path')

    # data section
    parser.add_argument('-label_set_path', '--label_set_path')
    parser.add_argument('-entities_path', '--entities_path')
    parser.add_argument('-embeddings_path', '--embeddings_path')
    parser.add_argument('-test_entities_path', '--test_entities_path')
    parser.add_argument('-test_embeddings_path', '--test_embeddings_path')

    # clustering setting
    parser.add_argument('--enable_cuda', action='store_true')
    parser.add_argument('-rate', '--rate', default=0.01)
    parser.add_argument('-mode', '--mode')
    parser.add_argument('-probing_cluster_path', '--probing_cluster_path')

    args = parser.parse_args()
    return args
