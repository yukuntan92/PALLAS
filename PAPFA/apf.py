#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
from .update_mat import *
from .net_model import *

def apf(N, all_poss_state, model, observation, dim_unk, num_gene, unk, bias, lam, num_sample):
    '''
    Input:
            N: number of particles
            all_poss_state: a two dimensions vector which contains all Boolean state combinations
            model: a list which contains -> [noise, data_type, baseline, delta, variance, depth]
            observation: the input data
            dim_unk: a list contains the number of search dimensions -> [the number of unknown discrete parameters, the number of unknown noise (1), the number of unknown baseline, the number of unknown delta, the number of unknown variance]
            num_gene: the number of genes in the network
            unk: the estimated value from fss.py
            bias: a system parameter which is a list of -1/2 when there is no damage information on the gene
            lam: the parameter of regularization term
            num_sample: the sample size of the given data

    Output:
            the probability of the esitmated value which given by fss.py based on the input dataset

    '''
    next_state, net_connection = net_model(all_poss_state, dim_unk, num_gene, unk, bias)
    match_state = np.dot(next_state, 1 << np.arange(num_gene-1, -1, -1))
    model[0] = unk[dim_unk[0]]
    model[2] = unk[dim_unk[0] + dim_unk[1] : dim_unk[0] + dim_unk[1] + dim_unk[2]]
    model[3] = unk[dim_unk[0] + dim_unk[1] + dim_unk[2] : dim_unk[0] + dim_unk[1] + dim_unk[2] + dim_unk[3]]
    model[4] = unk[dim_unk[0] + dim_unk[1] + dim_unk[2] + dim_unk[3] : dim_unk[0] + dim_unk[1] + dim_unk[2] + dim_unk[3] + dim_unk[4]]
    n = len(observation)
    if n % num_sample != 0:
        raise TypeError('missing value or incorrect sample size')
        sys.exit() 
    beta = 0
    for i in range(num_sample):
        init_prob = 1/2 ** (num_gene) * np.ones(2 ** num_gene)
        init_sample_state = np.random.multinomial(N, init_prob)  # initial state
        state_pos = init_sample_state.nonzero()[0]
        state_count = init_sample_state[state_pos]
        norm_wgt = 1/N * np.ones(N)  # initial weight
        for j in range(int(n / num_sample)):
            mu = match_state[state_pos].repeat(state_count)
            update_q = update_matrix(observation[num_sample * i + j], all_poss_state[mu,], model)  # p(Y_k | f(x_{k - 1}))
            if not np.any(update_q):
                update_q = np.repeat(1e-320, N)
            fst_wgt = update_q * norm_wgt  # first-stage weight or presampling probability
            if not np.any(fst_wgt):
                fst_wgt = np.repeat(1/N, N)
            norm_fst_wgt = fst_wgt / sum(fst_wgt)
            sample_state = np.random.multinomial(N, norm_fst_wgt)
            state_pos = sample_state.nonzero()[0]
            state_count = sample_state[state_pos]
            fst_wgt_dist = np.repeat(state_pos, state_count)  # discrete distribution of first-stage weight
            dist_mu = mu[state_pos].repeat(state_count)
            new_particle = ((all_poss_state[dist_mu,] + (np.random.uniform(0, 1, N * num_gene).reshape(N, num_gene) < model[0]) * 1) == 1) * 1  # new particles
            new_state_match = np.dot(new_particle, 1 << np.arange(num_gene-1, -1, -1))
            update_p = update_matrix(observation[num_sample * i + j], new_particle, model)  # update matrix with new particles
            if not np.any(update_p):
                update_p = np.repeat(1e-320, N)
            if np.any(update_q < 1e-10):
                update_q = update_q + 1e-10
            sec_wgt = update_p / update_q[fst_wgt_dist]  # second-stage weight
            if (np.mean(sec_wgt) * np.mean(fst_wgt)) == 0:
                beta += math.log(1e-320)
            else:
                beta += math.log(np.mean(sec_wgt) * np.mean(fst_wgt))  # the likelihood is the sample mean of the first-stage weight times the sample mean of the second-stage weight
            if not np.any(sec_wgt):
                sec_wgt = np.repeat(1/N, N)
            norm_wgt = sec_wgt / sum(sec_wgt)
            count = np.bincount(new_state_match)
            state_pos = count.nonzero()[0]
            state_count = count[state_pos]
    beta = beta / n - lam * sum(sum(abs(net_connection)))
    return beta

