#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def net_model(state, dim_unk, num_gene, unk, bias):
    '''
    return the next state Boolean vector based on current state vector and network
    '''
    net_connection = unk[:dim_unk[0]].reshape((num_gene, num_gene))
    net = np.c_[net_connection, bias]
    next_state = (np.dot(net, np.c_[state, np.ones(2**num_gene)].T) > 0) * 1
    return next_state.T, net_connection
