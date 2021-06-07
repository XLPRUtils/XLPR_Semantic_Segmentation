#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:2021/6/4 10:08
# @Author:Jianyuan Hong
# @File:__init__.py.py
# @Software:PyCharm

"""
This module provides data loaders and transformers for popular vision datasets.
"""
from .FUSeg import FootUlcerSegmentation




datasets = {
    'fuseg': FootUlcerSegmentation,
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
