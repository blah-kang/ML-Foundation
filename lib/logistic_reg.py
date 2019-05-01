import numpy as np
import sys
if "../" not in sys.path:
  sys.path.append("../") 
from lib.utils import read_data_from_file, sign, sigmoid


def logistic_reg(X, y, eta=0.1, epsilon=1e-6, updates=5000):
    """
    Logistic Regression Algorithm
    Args:
        X: 数据
        Y: 标签
        eta: 步长
        epsilon: 误差
        updates: 迭代次数
    
    Returns:
        w: 特征权重
        gradient: 梯度 
    """
    w = np.zeros_like(X[0])
    gradient = np.zeros_like(w)
    
    for i in range(1, updates + 1):
        if i%10 == 0:
            sys.stdout.flush()
            print('\repisode: {}'.format(i), end='')
        
        # step1: cal gradient
        gradient = np.mean((sigmoid(X.dot(w) * (-y))* -y).reshape((-1, 1))*X, axis=0)
        
        # step2: update weight
        w = w - eta*gradient
        
        if np.linalg.norm(gradient) <= epsilon:
            break
    return w, gradient
    
    
    
def logistic_reg_sgd(X, y, eta=0.05, epsilon=1e-6, updates=20000, random_choice=True):
    """
    Logistic Regression Stochastic Gradient Descent Algorithm
    Args:
        X: 数据
        Y: 标签
        eta: 步长
        epsilon: 误差
        updates: 迭代次数
    
    Returns:
        w: 特征权重
        gradient: 梯度
    """    
    w = np.zeros_like(X[0])
    gradient = np.zeros_like(w)
    N = len(X)
    
    for i in range(1, updates + 1):
        if i%50 == 0:
            sys.stdout.flush()
            print('\repisode: {}\t'.format(i), end='')
        
        # step1: random pick up one example
        if random_choice:
            index = np.random.choice(range(len(X)))
        else:
            index = (i-1)%N
        
        # step1: cal sgd
        gradient = (sigmoid(X[index].dot(w) * (-y[index]))* -y[index])*X[index]
        
        # step2: update weight
        w = w - eta*gradient
        
        if np.linalg.norm(gradient) <= epsilon or i>=updates:
            break
    return w, gradient