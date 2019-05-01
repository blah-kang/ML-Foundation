# -*- coding: utf-8 -*-
"""
Author  : kang <pujk2016@gmail.com>
Date    : 2019-4-20
"""

import numpy as np
import sys
if "../" not in sys.path:
  sys.path.append("../") 
from lib.utils import sign


def pla(X, y, eta=1):
    """
    Perceptron Learning Algorithm
    Args:
        X: 数据
        Y: 标签
        alpha: 步长
    
    Returns:
        w: 特征权重
        updates: 更新次数 
    """
    N = len(X)                               # examples size
    updates = 0                              
    pos = 0                                  # position of last correction mistake 
    w = np.zeros_like(X[0])                  
    
    for i in itertools.count():
        indx = i%N
        if sign(w.dot(X[indx]))*y[indx] < 0:
            w = w + X[indx]*y[indx] * eta      # (try to) correct the mistake
            updates += 1
            pos = i
        if i - pos >= N:
            break
    return w, updates


def pocket_pla(X, y, updates=50, w0=None):
    """
    Pocket Perceptron Learning Algorithm
    Args:
        X: 数据
        Y: 标签
        updates: 更新次数
        w0: 初始值
    Returns:
        w_pocket: 最优特征权重
        w: 最后更新得到的特征权重
    """
    w = np.zeros_like(X[0]) if w0 is None else w0
    w_pocket = w
    mistakes = np.where(sign(X.dot(w)) != y)[0]         # get index of mistakes
    mis_pocket = len(mistakes)
    
    for i in range(updates):
        mistake = np.random.choice(mistakes)            # pike up one mistake randomly
        w = w + X[mistake]*y[mistake]
        mistakes = np.where(sign(X.dot(w)) != y)[0]
        if mis_pocket > len(mistakes): 
            w_pocket = w
            mis_pocket = len(mistakes)
    return w_pocket, w