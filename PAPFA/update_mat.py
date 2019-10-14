#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import gammaln

def update_matrix(observation, boolean_states, model):
    '''
    return the measurement density (update matrix)
    '''
    model[2] = np.array(model[2])
    model[3] = np.array(model[3])
    model[4] = np.array(model[4])
    observation = np.array(observation)
    if model[1] == 'Gaussian':   # Gaussian distribution -> microarray data
        if min(model[4]) < 0.01:
            model[4] = model[4] + 0.01
        pro_0 = (1 / np.sqrt(2 * np.pi * model[4])) * np.exp(-(observation - model[2]) ** 2 / (2 * model[4]))
        pro_1 = (1 / np.sqrt(2 * np.pi * model[4])) * np.exp(-(observation - model[3] - model[2]) ** 2 / (2 * model[4]))
        T = (boolean_states == 0) * pro_0 + (boolean_states == 1) * pro_1
        T = np.prod(T, axis = 1)
    if model[1] == 'NB':   # Negative binomial distribution -> RNA-Seq data
        loga = np.exp(gammaln(observation + model[4]) - gammaln(observation + np.ones(len(observation))) - gammaln(model[4]))
        lam_0 = model[5] * np.exp(model[2])
        lam_1 = model[5] * np.exp(model[2] + model[3])
        pro_0 = loga * np.exp(observation * np.log(lam_0 / (lam_0 + model[4])) + model[4] * np.log(model[4] / (lam_0 + model[4])))
        pro_1 = loga * np.exp(observation * np.log(lam_1 / (lam_1 + model[4])) + model[4] * np.log(model[4] / (lam_1 + model[4])))
        T = (boolean_states == 0) * pro_0 + (boolean_states == 1) * pro_1
        T = np.prod(T, axis = 1)
    return T

