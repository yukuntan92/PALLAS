#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
from .generative_model import *
from .net_model import *
import time

def abcsmc(N, all_poss_state, model, observation, dim_unk, num_gene, unk, lam, num_sample, known_net, known_bias, M, tolerance):
    unk_net = unk[:dim_unk[0]]
    unk_bias = unk[dim_unk[0] : dim_unk[0] + dim_unk[1]]
    model[0] = unk[dim_unk[0] + dim_unk[1]]
    model[2] = unk[dim_unk[0] + dim_unk[1] + dim_unk[2] : dim_unk[0] + dim_unk[1] + dim_unk[2] + dim_unk[3]]
    model[3] = unk[dim_unk[0] + dim_unk[1] + dim_unk[2] + dim_unk[3] : dim_unk[0] + dim_unk[1] + dim_unk[2] + dim_unk[3] + dim_unk[4]]
    model[4] = unk[dim_unk[0] + dim_unk[1] + dim_unk[2] + dim_unk[3] + dim_unk[4] : dim_unk[0] + dim_unk[1] + dim_unk[2] + dim_unk[3] + dim_unk[4] + dim_unk[5]]
    model[5] = unk[dim_unk[0] + dim_unk[1] + dim_unk[2] + dim_unk[3] + dim_unk[4] + dim_unk[5] : dim_unk[0] + dim_unk[1] + dim_unk[2] + dim_unk[3] + dim_unk[4] + dim_unk[5] + dim_unk[6]]
    next_state, net_connection = net_model(all_poss_state, dim_unk, num_gene, unk_net, unk_bias, known_net, known_bias)
    match_state = np.dot(next_state, 1 << np.arange(num_gene-1, -1, -1))
    n = len(observation)
    if n % num_sample != 0:
        raise TypeError('missing value or incorrect sample size')
        sys.exit() 
    beta = 0
    init_prob = 1/2 ** (num_gene) * np.ones(2 ** num_gene)
    init_sample_state = np.random.multinomial(N, init_prob)
    state_pos = init_sample_state.nonzero()[0]
    state_count = init_sample_state[state_pos]
    norm_wgt = 1/N * np.ones(N)
    err = np.logspace(np.log10(tolerance[1]), np.log10(tolerance[0]), n)
    #err = np.linspace(tolerance[1], tolerance[0], n)
    record = []
    for i in range(n):
        mu = match_state[state_pos].repeat(state_count)
        new_particle = ((all_poss_state[mu,] + (np.random.uniform(0, 1, N * num_gene).reshape(N, num_gene) < model[0]) * 1) == 1) * 1  # new particles
        temp = generate(observation[i], new_particle, model, M, err[i], N)
        record.append(temp)
        norm_wgt *= temp
        #if (np.mean(norm_wgt)) == 0:
        if (np.mean(temp)) == 0:
            beta += math.log(1e-320)
        else:
            #beta += math.log(np.mean(norm_wgt))
            beta += math.log(np.mean(temp))
        if not np.any(norm_wgt):
            norm_wgt = np.repeat(1e-320, N)
        norm_wgt = norm_wgt/sum(norm_wgt)
        if (1/sum(norm_wgt ** 2)) < N/2:
            sample_state = np.random.multinomial(N, norm_wgt)
            state_pos = sample_state.nonzero()[0]
            state_count = sample_state[state_pos]
            new_particle = new_particle[state_pos].repeat(state_count, axis=0)
            norm_wgt = 1/N * np.ones(N)
        new_state_match = np.dot(new_particle, 1 << np.arange(num_gene-1, -1, -1))
        sample_state = np.bincount(new_state_match)
        state_pos = sample_state.nonzero()[0]
        state_count = sample_state[state_pos]
    beta = beta / n - lam * sum(sum(abs(net_connection)))
    return beta, record

