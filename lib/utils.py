# -*- coding: utf-8 -*-
"""
Author  : kang <pujk2016@gmail.com>
Date   : 2019-4-20
"""

import numpy as np
import re


def sign(x):
    return -1 if x<=0 else 1

sign = np.frompyfunc(sign, 1, 1)

sigmoid = lambda s: 1/(1+np.exp(-s))

def read_data_from_file(fpath):
    data = []
    with open(fpath, 'r') as f:
        for line in f:
            data.append(re.split(' |\t', line.strip()))
    data = np.array(data, dtype=np.float64)
    return data