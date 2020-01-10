#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def net_model(state, dim_unk, num_gene, unk_net, unk_bias, known_net, known_bias):
    '''
    return the next state Boolean vector based on current state vector and network
    '''
    if known_net is not None:
        for i in range(len(known_net)):
            unk_net = np.insert(unk_net, int(((known_net[i][1]-1)*num_gene + known_net[i][0]-1)), known_net[i][2]) 
    if known_bias is not None:
        for i in range(len(known_bias)):
            unk_bias = np.insert(unk_bias, int(known_bias[i][0]-1), known_bias[i][1])
    net_connection = unk_net.reshape((num_gene, num_gene))
    net = np.c_[net_connection, unk_bias]
    next_state = (np.dot(net, np.c_[state, np.ones(2**num_gene)].T) > 0) * 1
    return next_state.T, net_connection
