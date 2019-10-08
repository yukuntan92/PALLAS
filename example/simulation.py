#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math

def simulate(X, model, net):
    X_new = (((np.dot(net.T, np.r_[X, 1]) > 0) * 1 + (np.random.uniform(0,1,len(X)) < model[0])*1) == 1) * 1
    if model[1] == "NB":
        observation = []
        for i in range(len(net)-1):
            lamm = model[2]*math.exp(model[3][i] + model[4][i]*X_new[i])
            observation.extend(list(np.random.negative_binomial(n=model[5][i], p=model[5][i]/(model[5][i]+lamm), size=1)))
        return X_new, observation
    if model[1] == "Gaussian":
        observation = model[2] * np.ones(len(net)-1) + np.dot(np.eye(len(net)-1, dtype=int), X_new) * model[3] + np.random.normal(0, math.sqrt(model[4]), len(net)-1)
        return X_new, observation

net_connection = np.array([[0,0,-1,0],[1,0,-1,-1],[0,1,0,0],[-1,1,1,0]])
bias = np.array([-1/2,-1/2,-1/2,-1/2]) + np.array([1,0,0,0]) 
net = np.vstack((net_connection.T,bias))
X = [0,1,0,1]
model = [0.01, 'Gaussian', 30, 20, 49]
#model = [0.01, 'NB', 22.52, [0.1] * 4, [2] * 4, [5] * 4]
with open('ga_data.txt', 'w') as f:
    f.write('A\tB\tC\tD\n')
    for _ in range(50):
        [X, observation] = simulate(X, model, net)
        if model[1] == 'NB':
            observation = str(observation)[1:-1].split(',')
        else:
            observation = str(observation)[1:-1].split()
        f.write(observation[0]+'\t'+observation[1]+'\t'+observation[2]+'\t'+observation[3]+'\n')

