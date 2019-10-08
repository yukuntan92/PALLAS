#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
from update_mat import update_matrix
from net_model import net_model
from collections import Counter
import time

def apf(N, all_poss_state, model, observation, dim_unk, num_gene, unk, bias, lam, num_sample):
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
        init_sample_state = np.random.multinomial(N, init_prob)
        state_pos = init_sample.nonzero()[0]
        state_count = init_sample[sample_pos]
        wgt = 1/N * np.ones(N)
        for j in range(int(n / num_sample)):
            mu = match_state[state_pos].repeat(state_count)
            update_pr = update_matrix(observation[num_sample * i + j], all_poss_state[mu,], model)
            if not np.any(update_pr):
                update_pr = np.repeat(1e-320, N)
            prw = update_pr * wgt
            if not np.any(prw):
                prw = np.repeat(1/N, N)
            prw = prw / sum(prw)
            sample_state = np.random.multinomial(N, prw)
            state_pos = sample.nonzero()[0]
            state_count = sample[sample_pos]
            Sx = np.repeat(sample_pos, sample_count)
            zz = mu[sample_pos].repeat(sample_count)
            xkc = ((all_poss_state[zz,] + (np.random.uniform(0, 1, N * num_gene).reshape(N, num_gene) < model[0]) * 1) == 1) * 1
            new_state_match = np.dot(xkc, 1 << np.arange(num_gene-1, -1, -1))
            prdT = update_matrix(observation[num_sample * i + j], xkc, model)
            if not np.any(prdT):
                prdT = np.repeat(1e-320, N)
            if np.any(update_pr < 1e-10):
                update_pr = update_pr + 1e-10
            w_k = prdT / update_pr[Sx]
            if (np.mean(w_k) * np.mean(update_pr)) == 0:
                beta += math.log(1e-320)
            else:
                beta += math.log(np.mean(w_k) * np.mean(update_pr))
            if not np.any(w_k):
                w_k = np.repeat(1/N, N)
            wgt = w_k / sum(w_k)
            count_state = np.bincount(new_state_match)
            state_pos = count_state.nonzero()[0]
            state_count = sample[state_pos]
    constr = sum(abs(net_connection).T)
    beta = beta / n - lam * sum(constr)
    return beta

