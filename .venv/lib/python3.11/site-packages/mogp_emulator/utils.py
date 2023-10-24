# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 05:32:45 2013

@author: tisimst
"""
import random
from copy import copy
import numpy as np
import scipy.stats as ss

def k_fold_cross_validation(X, K, randomise=False):
    """
    Generates K (training, validation) pairs from the items in X.

    Each pair is a partition of X, where validation is an iterable
    of length len(X)/K. So each training iterable is of length (K-1)*len(X)/K.

    If randomise is true, a copy of X is shuffled before partitioning,
    otherwise its order is preserved in training and validation.
    ## {{{ http://code.activestate.com/recipes/521906/ (r3)
    """
    if randomise:
        X = list(X)
        random.shuffle(X)
    for k in range(K):
        training = [x for i, x in enumerate(X) if i % K != k]
        validation = [x for i, x in enumerate(X) if i % K == k]
        yield training, validation

        
def integer_bisect(bound, f):
    """
    Finds a pair of integers (a,b) such that f(a) <= 0 < f(b) and |a - b| == 1.
    On entry, assumes that f(bound[0]) <= 0 < f(bound[1])
    """
    if bound[1] - bound[0] == 1:
        return bound
    else:
        midpoint = round(bound[0] + bound[1] / 2.0)
        if f(midpoint) <= 0:
            return integer_bisect((midpoint, bound[1]), f)
        else:
            return integer_bisect((bound[0], midpoint), f)
