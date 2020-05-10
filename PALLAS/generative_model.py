#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
import concurrent.futures

#def generate_data(observation, boolean_state, model, M, err):
#    gamma = np.random.gamma(model[2], model[3], len(observation))
#    gam = (model[5] - 1) * boolean_state * gamma + gamma
#    generate = np.random.multivariate_normal(gam, np.diag(gamma**2*model[4]), M)
#    weight = np.sum(np.sqrt(np.sum((generate - observation)**2, axis=1)) < err)/M
#    return weight
#
#def wrapper(p):
#    return generate_data(*p)
#
#def generate(observation, boolean_state, model, M, err):
#    model[2] = np.array(model[2])
#    model[3] = np.array(model[3])
#    model[4] = np.array(model[4])
#    model[5] = np.array(model[5])
#    observation = np.array(observation)
#    weight = []
#    #with concurrent.futures.ThreadPoolExecutor() as executor:
#    with concurrent.futures.ProcessPoolExecutor() as executor:
#        args = [(observation, boolean_state[i], model, M, err) for i in range(16)]
#        weight = executor.map(wrapper, args)
#    weight = list(weight)

def generate(observation, boolean_state, model, M, err, N):
    model[2] = np.array(model[2])
    model[3] = np.array(model[3])
    model[4] = np.array(model[4])
    model[5] = np.array(model[5])
    observation = np.array(observation)
    weight = []
    for i in range(N):
        gamma = np.random.gamma(model[2], model[3], len(observation))
        gam = (model[5] - 1) * boolean_state[i] * gamma + gamma
        generate = np.random.multivariate_normal(gam, np.diag(gamma**2*model[4]), M)
        weight.append(np.sum(np.sqrt(np.sum((generate - observation)**2, axis=1)) < err)/M)
    return np.array(weight)

