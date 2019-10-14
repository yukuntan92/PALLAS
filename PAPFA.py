#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sys
from itertools import product
from PAPFA.fss import *

argumentsValues = {'input':'', 'data_type':'', 'noise':[0, 0.1], 'baseline':'', 'delta':'', 'variance':'', 'diff_baseline':False, 'diff_delta':False, 'diff_variance':False, 'fish':'', 'iteration':5000, 'lambda':0.01, 'particle':'', 'depth':22.52, 'damage':False, 'sample':1}

def main(argv=sys.argv):
    for arg in sys.argv:
        if '.py' not in arg and '=' not in arg:
            raise TypeError('incorrect argument '+arg)
            break
        if '.py' not in arg:
            name, val = arg.split('=')
            if name == 'input':
                argumentsValues[name] = val
            elif name == 'data_type':
                if val == 'rnaseq':
                    argumentsValues[name] = 'NB'
                elif val == 'microarray':
                    argumentsValues[name] = 'Gaussian'
                else:
                    raise TypeError('incorrect argument '+arg)
                    break
            elif name == 'noise' or name == 'baseline' or name == 'delta' or name == 'variance':
                argumentsValues[name] = list(val.split('-'))
            elif name == 'fish' or name == 'iteration' or name == 'particle' or name == 'sample':
                argumentsValues[name] = int(val)
            elif name == 'lambda' or name == 'depth':
                argumentsValues[name] = float(val)
            elif name == 'diff_baseline' or name == 'diff_delta' or name == 'diff_variance':
                if val == 'True':
                    argumentsValues[name] = True
                else:
                    argumentsValues[name] = False
            elif name == 'damage':
                if val == 'False':
                    argumentsValues[name] = False
                else:
                    argumentsValues[name] = int(val)

    data = []
    if argumentsValues['data_type'] == 'NB':
        with open(argumentsValues['input'], 'r') as f:
            next(f)
            for line in f:
                line_data = line.split('\t')
                l = [elem.rstrip() for elem in line_data]
                data.append(list(map(int, l)))
    else:
        with open(argumentsValues['input'], 'r') as f:
            next(f)
            for line in f:
                line_data = line.split('\t')
                l = [elem.rstrip() for elem in line_data]
                data.append([float(i) for i in l])

    data = np.array(data)
    num_gene = len(data[0])
    data_max = np.max(data)
    data_min = np.min(data)
    data_mean = np.mean(data)

    num_baseline = 1
    num_delta = 1
    num_variance = 1

    if argumentsValues['particle'] == '':
        argumentsValues['particle'] = 2 ** num_gene
    if argumentsValues['diff_baseline'] == True:
        num_baseline = num_gene
    if argumentsValues['diff_delta'] == True:
        num_delta = num_gene
    if argumentsValues['diff_variance'] == True:
        num_variance = num_gene
    if argumentsValues['fish'] == '':
        argumentsValues['fish'] = 3 * (num_gene ** 2 + num_baseline + num_delta + num_variance + 1)

    if argumentsValues['data_type'] == 'NB':
        data_min = data_min / argumentsValues['depth']
        data_max = data_max / argumentsValues['depth']
        data_mean = data_mean / argumentsValues['depth']
        if argumentsValues['baseline'] == '':
            if data_min < 3:
                argumentsValues['baseline'] = [0, np.log(data_mean)]
            else:
                argumentsValues['baseline'] = [np.log(data_min), np.log(data_mean)]
        if argumentsValues['delta'] == '':
            if data_min < 3:
                argumentsValues['delta'] = [min(np.log(data_max) - np.log(data_mean), np.log(data_mean)) / 3, np.log(data_max)]
            else:
                argumentsValues['delta'] = [min(np.log(data_max) - np.log(data_mean), np.log(data_mean) - np.log(data_min)) / 3, np.log(data_max) - np.log(data_min)]
        if argumentsValues['variance'] == '':
            argumentsValues['variance'] = [0.5, 8]
    else:
        if argumentsValues['baseline'] == '':
            argumentsValues['baseline'] = [data_min, data_mean]
        if argumentsValues['delta'] == '':
            argumentsValues['delta'] = [min(data_max - data_mean, data_mean - data_min) / 3, data_max - data_min]
        if argumentsValues['variance'] == '':
            argumentsValues['variance'] = [0.01, (max(data_max - data_mean, data_mean - data_min) / 3) ** 2]

    school_size = argumentsValues['fish']
    num_iterations = argumentsValues['iteration']
    num_sample = argumentsValues['sample']
    N = argumentsValues['particle']
    lam = argumentsValues['lambda']
    model = ["noise", argumentsValues['data_type'], ["baseline"] * num_baseline, ["delta"] * num_delta, ["variance"] * num_variance, argumentsValues['depth']]        # [noise, model, sequencing depth, baseline, delta, inverse dispersion]
    search_area = np.array([float(argumentsValues['noise'][0]), float(argumentsValues['noise'][1]), float(argumentsValues['baseline'][0]), float(argumentsValues['baseline'][1]), float(argumentsValues['delta'][0]), float(argumentsValues['delta'][1]), float(argumentsValues['variance'][0]), float(argumentsValues['variance'][1])])        # [noise_range, baseline_range, delta_range, variance_range]
    dim_unk = [num_gene ** 2, 1, num_baseline, num_delta, num_variance]       # dim_unk = [the number of unknown discrete parameter, the number of unknown continous parameter]
    
    inpt = np.zeros(num_gene)
    if argumentsValues['damage'] != False:
        inpt[argumentsValues['damage'] - 1,] = 1
    bias = -1/2 * np.ones(num_gene) + inpt

    if argumentsValues['data_type'] == 'NB':
        print(argumentsValues.items())
    else:
        argumentsValues.pop('depth')
        print(argumentsValues.items())
    
    all_poss_state = []
    for i in product([0.0, 1.0], repeat=num_gene):
        all_poss_state.append(i)
    all_poss_state = np.array(all_poss_state)
    [beta, unk, school] = fish_school_search(dim_unk, num_gene, bias, model, data, all_poss_state, school_size, num_iterations, N, lam, search_area, num_sample)
    print(unk)
    print('Source\tTarget\tInteraction\n')
    for i in range(num_gene ** 2):
        row = i%num_gene + 1
        col = i//num_gene + 1
        if np.allclose(unk[i], 1):
            print(str(row) + '\t'+str(col) + '\t' + 'activation' + '\n')
        elif np.allclose(unk[i], -1):
            print(str(row) + '\t' + str(col) + '\t' + 'inhibition' + '\n')
    noise = np.around(unk[num_gene ** 2], decimals=3)
    base = np.around(unk[num_gene ** 2 + 1 : num_gene ** 2 + 1 + num_baseline], decimals=2)
    delt = np.around(unk[num_gene ** 2 + 1 + num_baseline : num_gene ** 2 + 1 + num_baseline + num_delta], decimals=2)
    env_noise = np.around(unk[num_gene ** 2 + 1 + num_baseline + num_delta :], decimals=2)
    print('process noise = {}'.format(str(noise) + '\n'))
    print('baseline = {}'.format('\t'.join(map(str, base)) + '\n'))
    print('delta = {}'.format('\t'.join(map(str, delt)) + '\n'))
    print('environmental noise = {}'.format('\t'.join(map(str, env_noise)) + '\n'))

if __name__ == "__main__":
    main()
