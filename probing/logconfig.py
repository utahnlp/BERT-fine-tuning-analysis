# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: Yichu Zhou - flyaway1217@gmail.com
# Blog: zhouyichu.com
#
# Python release: 3.8.0
#
# Date: 2020-07-24 16:12:52
# Last modified: 2020-12-29 12:12:12

"""
Logger configurations.
"""

PACKAGE_NAME = 'probing'

LOG_FILE = PACKAGE_NAME + '.log'

LOGGING_CONFIG = {
     'version': 1,
     'disable_existing_loggers': True,
     'formatters': {
         'standard': {
             'format': '%(asctime)s-%(filename)s-%(levelname)s: %(message)s'
         },
     },
     'handlers': {
         'file': {
             'level': 'DEBUG',
             'formatter': 'standard',
             'class': 'logging.FileHandler',
             'filename': LOG_FILE,
         },
         'console': {
             'level': 'DEBUG',
             'formatter': 'standard',
             'class': 'logging.StreamHandler',
             # 'stream': 'ext://sys.stdout',
             },
     },
     'loggers': {
         PACKAGE_NAME: {
             'level': 'DEBUG',
             'handlers': ['console', 'file']
             },
         '__main__': {
             'level': 'DEBUG',
             'handlers': ['console', 'file']
             },
         },
 }


def set_log_path(path):
    LOGGING_CONFIG['handlers']['file']['filename'] = path
