#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sys
from itertools import product
from PALLAS.fss import *
from operator import itemgetter

argumentsValues = {'input':'', 'data_type':'', 'noise':[0.05, 0.05], 'baseline':'', 'delta':'', 'variance':'', 'diff_baseline':False, 'diff_delta':False, 'diff_variance':False, 'fish':'', 'iteration':5000, 'lambda':0.01, 'particle':'', 'depth':1.02, 'bias':False, 'net':False, 'sample':1, 'running_time':1, 'full_info':False}

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
                argumentsValues[name] = [float(i) for i in list(val.split('-'))]
            elif name == 'fish' or name == 'iteration' or name == 'particle' or name == 'sample' or name == 'running_time':
                argumentsValues[name] = int(val)
            elif name == 'lambda' or name == 'depth':
                argumentsValues[name] = float(val)
            elif name == 'diff_baseline' or name == 'diff_delta' or name == 'diff_variance' or name == 'full_info':
                if val == 'True':
                    argumentsValues[name] = True
                else:
                    argumentsValues[name] = False
            elif name == 'bias':
                if val == 'False':
                    argumentsValues[name] = False
                else:
                    argumentsValues[name] = val
            elif name == 'net':
                if val == 'False':
                    argumentsValues[name] = False
                else:
                    argumentsValues[name] = val

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
    data_max = np.max(data)
    data_min = np.min(data)
    data_mean = np.mean(data)
    num_gene = len(data[0])
   
    known_net = []
    if argumentsValues['net'] == False:
        num_net = num_gene ** 2
    else:
        with open(argumentsValues['net'], 'r') as f:
            next(f)
            n = 0
            for line in f:
                n += 1
                line_data = line.split('\t')
                l = [elem.rstrip() for elem in line_data]
                known_net.append([int(i) for i in l])
            num_net = num_gene ** 2 - n
            known_net = sorted(known_net, key=itemgetter(1, 0))

    known_bias = []
    if argumentsValues['bias'] == False:
        num_bias = num_gene
    else:
        with open(argumentsValues['bias'], 'r') as f:
            next(f)
            n = 0
            for line in f:
                n += 1
                line_data = line.split('\t')
                l = [elem.rstrip() for elem in line_data]
                known_bias.append([float(i) for i in l])
            num_bias = num_gene - n
            known_bias = sorted(known_bias, key=itemgetter(0))

    num_baseline = 1
    num_delta = 1
    num_variance = 1
    num_noise = 1

    if argumentsValues['particle'] == '':
        argumentsValues['particle'] = 2 ** num_gene
    if argumentsValues['diff_baseline'] == True:
        num_baseline = num_gene
    if argumentsValues['diff_delta'] == True:
        num_delta = num_gene
    if argumentsValues['diff_variance'] == True:
        num_variance = num_gene
    if argumentsValues['noise'][0] == argumentsValues['noise'][1]:
        num_noise = 0
    if argumentsValues['fish'] == '':
        argumentsValues['fish'] = 3 * (num_net + num_baseline + num_delta + num_variance + num_noise + num_bias)

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
            argumentsValues['variance'] = [0.5, 7]
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
    search_area = np.array([argumentsValues['noise'][0], argumentsValues['noise'][1], argumentsValues['baseline'][0], argumentsValues['baseline'][1], argumentsValues['delta'][0], argumentsValues['delta'][1], argumentsValues['variance'][0], argumentsValues['variance'][1]])        # [noise_range, baseline_range, delta_range, variance_range]
    dim_unk = [num_net, num_bias, 1, num_baseline, num_delta, num_variance]       # dim_unk = [the number of unknown discrete parameter, the number of unknown continous parameter]
    
    if argumentsValues['data_type'] == 'NB':
        print(argumentsValues.items())
    else:
        argumentsValues.pop('depth')
        print(argumentsValues.items())
    
    all_poss_state = []
    for i in product([0.0, 1.0], repeat=num_gene):
        all_poss_state.append(i)
    all_poss_state = np.array(all_poss_state)
    result = []
    for _ in range(argumentsValues['running_time']):
        [beta, unk, school] = fish_school_search(dim_unk, num_gene, model, data, all_poss_state, school_size, num_iterations, N, lam, search_area, num_sample, known_net, known_bias)
        result.append((beta, unk))
    result.sort(key=lambda x: (-x[0], sum(abs(x[1][:num_gene ** 2]))))
                
    for i in range(argumentsValues['running_time']):
        num = i + 1
        res = result[i][1]
        if known_net is not None:
            for j in range(len(known_net)):
                res = np.insert(res, int(((known_net[j][1]-1)*num_gene + known_net[j][0]-1)), known_net[j][2])
        if known_bias is not None:
            for j in range(len(known_bias)):
                res = np.insert(res, int(num_gene ** 2 + known_bias[j][0]-1), known_bias[j][1])
        print(res)
        print('Ranking' + ' ' + str(num))
        print('Source\tTarget\tInteraction\n')
        for j in range(num_gene ** 2):
            row = j%num_gene + 1
            col = j//num_gene + 1
            if np.allclose(res[j], 1):
                print(str(row) + '\t'+str(col) + '\t' + 'activation' + '\n')
            elif np.allclose(res[j], -1):
                print(str(row) + '\t' + str(col) + '\t' + 'inhibition' + '\n')
        bias = res[num_gene ** 2 : num_gene ** 2 + num_gene]
        noise = np.around(res[num_gene ** 2 + num_gene], decimals=3)
        base = np.around(res[num_gene ** 2 + num_gene + 1 : num_gene ** 2 + num_gene + 1 + num_baseline], decimals=2)
        delt = np.around(res[num_gene ** 2 + num_gene + 1 + num_baseline : num_gene ** 2 + num_gene + 1 + num_baseline + num_delta], decimals=2)
        env_noise = np.around(res[num_gene ** 2 + num_gene + 1 + num_baseline + num_delta :], decimals=2)
        likelihood = np.around(result[i][0], decimals=3)
        if argumentsValues['full_info']:
            print('bias = {}'.format(str(bias) + '\n'))
            print('process noise = {}'.format(str(noise) + '\n'))
            print('baseline = {}'.format('\t'.join(map(str, base)) + '\n'))
            print('delta = {}'.format('\t'.join(map(str, delt)) + '\n'))
            print('environmental noise = {}'.format('\t'.join(map(str, env_noise)) + '\n'))
            print('log-likelihood = {}'.format(str(likelihood) + '\n'))
        if noise in [float(argumentsValues['noise'][0]), float(argumentsValues['noise'][1])] and argumentsValues['noise'][0] != argumentsValues['noise'][1]:
            print('Warning! The estimated noise hit the boundary [', argumentsValues['noise'][0], ',', argumentsValues['noise'][1], '], please define a larger search space')
        if float(argumentsValues['baseline'][0]) in base or float(argumentsValues['baseline'][1]) in base:
            print('Warning! The estimated baseline hit the boundary [', argumentsValues['baseline'][0], ',', argumentsValues['baseline'][1], '], please define a larger search space')
        if float(argumentsValues['delta'][0]) in delt or float(argumentsValues['delta'][1]) in delt:
            print('Warning! The estimated delta hit the boundary [', argumentsValues['delta'][0], ',', argumentsValues['delta'][1], '], please define a larger search space')
        if float(argumentsValues['variance'][0]) in env_noise or float(argumentsValues['variance'][1]) in env_noise:
            print('Warning! The estimated variance hit the boundary [', argumentsValues['variance'][0], ',', argumentsValues['variance'][1], '], please define a larger search space')

if __name__ == "__main__":
    main()
