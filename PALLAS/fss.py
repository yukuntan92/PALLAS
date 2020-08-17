#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from .abcsmc import *
from .apf import *
from tqdm import tqdm

def individual_movement(best_fish, best_cost, cost, school, parameter_curr_step_individual, max_func, min_func, search_area, dim, num_gene, model, observation, thresh, all_poss_state, N, lam, num_sample, known_net, known_bias, M, tolerance):
    '''
    Input: 
            best_fish: the optimal position of the search space
            best_cost: the value of the fitness function of the best_fish
            cost: the best value of the fitness function currently
            school: a two dimension vector which contains all the candidate solutions (fishes)
            parameter_curr_step_individual: a list of the currently individual step based on continuous parameters
            max_func: 1 -> the upper boundary of discrete search space
            min_func: -1 -> the lower boundary of discrete search space
            search_area: a list contains all the search space of continuous parameters
            dim: the dimension of discrete and continuous parameters
            thresh: a criterion created in discrete fish school search algorithm
            num_gene, bias, model, observation, all_poss_state, N, lam, num_sample: the parameter used in apf.sh and described in apf.sh file
    Output: 
            update the best postion and value currently; obtain the displacement information of the fishes and the impovement of each fish because of this displacement
    '''
    total_dim = sum(dim)
    school_update_individual = []
    delta_cost_update_individual = []
    delta_position_update_individual = []
    for i in range(len(school)):
        new_position = np.zeros((total_dim,))
        random_step = np.zeros((total_dim,))
        for j in range(total_dim):  # generate random step for all the dimensions in fish
            if j < dim[0] + dim[1]:
                random_step[j] = np.random.uniform(-1, 1)
            elif j < dim[0] + dim[1] + dim[2]:
                random_step[j] = parameter_curr_step_individual[0] * np.random.uniform(-1, 1)
            elif j < dim[0] + dim[1] + dim[2] + dim[3]:
                random_step[j] = parameter_curr_step_individual[1] * np.random.uniform(-1, 1)
            elif j < dim[0] + dim[1] + dim[2] + dim[3] + dim[4]:
                random_step[j] = parameter_curr_step_individual[2] * np.random.uniform(-1, 1)
            elif j < dim[0] + dim[1] + dim[2] + dim[3] + dim[4] + dim[5]:
                random_step[j] = parameter_curr_step_individual[3] * np.random.uniform(-1, 1)
            elif model[1] == 'mixed' and j < total_dim:
                random_step[j] = parameter_curr_step_individual[4] * np.random.uniform(-1, 1)
        if all(k > 0 for k in random_step[0:dim[0]]):  # the criterion to discretized the moving for discrete dimensions
            pos_threshold = max(random_step[0:dim[0]]) * thresh
            neg_threshold = 0
        elif all(k < 0 for k in random_step[0:dim[0]]):
            pos_threshold = 0
            neg_threshold = min(random_step[0:dim[0]]) * thresh
        else:
            pos_threshold = max(random_step[0:dim[0]]) * thresh
            neg_threshold = min(random_step[0:dim[0]]) * thresh
        for j in range(dim[0]):
            if random_step[j] > pos_threshold:
                random_step[j] = 1
            elif random_step[j] < neg_threshold:
                random_step[j] = -1
            else:
                random_step[j] = 0
        if dim[1] != 0:
            if all(k > 0 for k in random_step[dim[0]:dim[0]+dim[1]]):
                pos_bias = max(random_step[dim[0]:dim[0]+dim[1]]) * thresh
                neg_bias = 0
            elif all(k < 0 for k in random_step[dim[0]:dim[0]+dim[1]]):
                pos_bias = 0
                neg_bias = min(random_step[dim[0]:dim[0]+dim[1]]) * thresh
            else:
                pos_bias = max(random_step[dim[0]:dim[0]+dim[1]]) * thresh
                neg_bias = min(random_step[dim[0]:dim[0]+dim[1]]) * thresh
            for j in range(dim[0], dim[0]+dim[1]):
                if random_step[j] > pos_bias:
                    random_step[j] = 1
                elif random_step[j] < neg_bias:
                    random_step[j] = -1
                else:
                    random_step[j] = 0
        for j in range(total_dim):
            new_position[j] = school[i][j] + random_step[j]   # displacement of the fish and handle the case when moving out of the boundary
            if j < dim[0]:
                if new_position[j] < min_func:
                    new_position[j] = min_func
                elif new_position[j] > max_func:
                    new_position[j] = max_func
            elif j < dim[0] + dim[1] and dim[1] != 0:
                if new_position[j] < -1/2:
                    new_position[j] = -1/2
                elif new_position[j] > 1/2:
                    new_position[j] = 1/2
            elif j < dim[0] + dim[1] + dim[2]:
                if new_position[j] < search_area[0]:
                    new_position[j] = search_area[0] 
                elif new_position[j] > search_area[1]:
                    new_position[j] = search_area[1] 
            elif j < dim[0] + dim[1] + dim[2] + dim[3]:
                if new_position[j] < search_area[2]:
                    new_position[j] = search_area[2]
                elif new_position[j] > search_area[3]:
                    new_position[j] = search_area[3]
            elif j < dim[0] + dim[1] + dim[2] + dim[3] + dim[4]:
                if new_position[j] < search_area[4]:
                    new_position[j] = search_area[4]
                elif new_position[j] > search_area[5]:
                    new_position[j] = search_area[5]
            elif j < dim[0] + dim[1] + dim[2] + dim[3] + dim[4] + dim[5]:
                if new_position[j] < search_area[6]:
                    new_position[j] = search_area[6]
                elif new_position[j] > search_area[7]:
                    new_position[j] = search_area[7]
            elif model[1] == 'mixed' and j < total_dim:
                if new_position[j] < search_area[8]:
                    new_position[j] = search_area[8]
                elif new_position[j] > search_area[9]:
                    new_position[j] = search_area[9]
        if model[1] == 'mixed':
            curr_cost, record = abcsmc(N, all_poss_state, model, observation, dim, num_gene, new_position, lam, num_sample, known_net, known_bias, M, tolerance)
            #curr_cost = abcsmc(N, all_poss_state, model, observation, dim, num_gene, new_position, lam, num_sample, known_net, known_bias, M, tolerance)
        else:
            curr_cost = apf(N, all_poss_state, model, observation, dim, num_gene, new_position, lam, num_sample, known_net, known_bias)   # calculate the fitness function
        if curr_cost > cost[i]:  # update the currently optimal position and fitness function value
            delta_cost = curr_cost - cost[i]
            cost[i] = curr_cost
            delta_position = np.zeros((total_dim,))
            delta_position = new_position - school[i]
            school_update_individual.append(new_position) 
            if curr_cost > best_cost:
                best_cost = curr_cost
                best_fish = new_position
        else:
            delta_position = np.zeros((total_dim,))
            delta_cost = 0
            school_update_individual.append(school[i])
        delta_cost_update_individual.append(delta_cost)
        delta_position_update_individual.append(delta_position)
    if model[1] == 'mixed':
        return best_fish, best_cost, school_update_individual, cost, delta_position_update_individual, delta_cost_update_individual, record
    else:
        return best_fish, best_cost, school_update_individual, cost, delta_position_update_individual, delta_cost_update_individual


def feeding(weight, max_weight, min_weight, delta_cost):
    '''
    Input:
            weight: a variable to measure how good the fish's position is
            delta_cost: fitness impovement from individual movement
    Output:
            updated weight
    '''
    max_delta = max(delta_cost)
    for i in range(len(delta_cost)):
        if max_delta:
            weight[i] += (delta_cost[i] / max_delta)
        if weight[i] > max_weight:
            weight[i] = max_weight
        elif weight[i] < min_weight:
            weight[i] = min_weight
    return weight

def collective_instinctive_movement(school, dim, delta_position, delta_cost, min_func, max_func, search_area, thresh, model):
    '''
    Input: 
            delta_position: displacement from individual movement
    Output:
            update the fishes position based on the fishes which had successful individual movements
    '''
    total_dim = sum(dim)
    instinctive_step = np.zeros((total_dim,))
    density = 0.0
    school_update_instinctive = []
    for i in range(len(school)):
        density += delta_cost[i]
        instinctive_step += (delta_position[i] * delta_cost[i])
    if density:
        instinctive_step = instinctive_step / density
    if all(k > 0 for k in instinctive_step[0:dim[0]]):
        pos_threshold = max(instinctive_step[0:dim[0]]) * thresh
        neg_threshold = 0
    elif all(k < 0 for k in instinctive_step[0:dim[0]]):
        pos_threshold = 0
        neg_threshold = min(instinctive_step[0:dim[0]]) * thresh
    else:
        pos_threshold = max(instinctive_step[0:dim[0]]) * thresh
        neg_threshold = min(instinctive_step[0:dim[0]]) * thresh
    for i in range(dim[0]):
        if instinctive_step[i] > pos_threshold:
            instinctive_step[i] = 1
        elif instinctive_step[i] < neg_threshold:
            instinctive_step[i] = -1
        else:
            instinctive_step[i] = 0
    if dim[1] != 0:
        if all(k > 0 for k in instinctive_step[dim[0]:dim[0]+dim[1]]):
            pos_bias = max(instinctive_step[dim[0]:dim[0]+dim[1]]) * thresh
            neg_bias = 0
        elif all(k < 0 for k in instinctive_step[dim[0]:dim[0]+dim[1]]):
            pos_bias = 0
            neg_bias = min(instinctive_step[dim[0]:dim[0]+dim[1]]) * thresh
        else:
            pos_bias = max(instinctive_step[dim[0]:dim[0]+dim[1]]) * thresh
            neg_bias = min(instinctive_step[dim[0]:dim[0]+dim[1]]) * thresh
        for i in range(dim[0], dim[0]+dim[1]):
            if instinctive_step[i] > pos_bias:
                instinctive_step[i] = 1
            elif instinctive_step[i] < neg_bias:
                instinctive_step[i] = -1
            else:
                instinctive_step[i] = 0
    for fish in school:
        new_position = np.zeros((total_dim,))
        for i in range(total_dim):
            new_position[i] = fish[i] + instinctive_step[i]
            if i < dim[0]:
                if new_position[i] < min_func:
                    new_position[i] = min_func
                elif new_position[i] > max_func:
                    new_position[i] = max_func
            elif i < dim[0] + dim[1] and dim[1] != 0:
                if new_position[i] < -1/2:
                    new_position[i] = -1/2
                elif new_position[i] > 1/2:
                    new_position[i] = 1/2
            elif i < dim[0] + dim[1] + dim[2]:
                if new_position[i] < search_area[0]:
                    new_position[i] = search_area[0] 
                elif new_position[i] > search_area[1]:
                    new_position[i] = search_area[1] 
            elif i < dim[0] + dim[1] + dim[2] + dim[3]:
                if new_position[i] < search_area[2]:
                    new_position[i] = search_area[2]
                elif new_position[i] > search_area[3]:
                    new_position[i] = search_area[3]
            elif i < dim[0] + dim[1] + dim[2] + dim[3] + dim[4]:
                if new_position[i] < search_area[4]:
                    new_position[i] = search_area[4]
                elif new_position[i] > search_area[5]:
                    new_position[i] = search_area[5]
            elif i < dim[0] + dim[1] + dim[2] + dim[3] + dim[4] + dim[5]:
                if new_position[i] < search_area[6]:
                    new_position[i] = search_area[6]
                elif new_position[i] > search_area[7]:
                    new_position[i] = search_area[7]
            elif model[1] == 'mixed' and i < total_dim:
                if new_position[i] < search_area[8]:
                    new_position[i] = search_area[8]
                elif new_position[i] > search_area[9]:
                    new_position[i] = search_area[9]
        school_update_instinctive.append(new_position)
    return school_update_instinctive

def calculate_barycenter(weight, dim, school):
    '''
    calculate the barycenter of fish school
    '''
    barycenter = np.zeros((sum(dim),))
    density = 0.0
    for i in range(len(school)):
        density += weight[i]
        barycenter += (school[i] * weight[i])
    barycenter = barycenter / density
    return barycenter

def collective_volitive_movement(best_fish, best_cost, curr_weight_school, weight, dim, school, min_func, max_func, search_area, parameter_curr_step_volitive, num_gene, model, observation, thresh, all_poss_state, N, lam, num_sample, known_net, known_bias, M, tolerance):
    '''
    Input:
            parameter_curr_step_volitive: step size of volitive movement which equal to twice of individual step size
    Output:
            the best position, the best fitness value and current fishes position, weight and cost updated by volitive movement

    In this movement, the radius of fish school will contract if total weight of fish school increases, otherwise the fish school expand
    '''
    total_dim = sum(dim)
    prev_weight_school = curr_weight_school
    curr_weight_school = sum(weight)
    barycenter = calculate_barycenter(weight, dim, school)
    school_update_volitive = []
    cost = []
    for fish in school:
        new_position = np.zeros((total_dim,))
        step = np.zeros((total_dim,))
        for j in range(total_dim):
            if j < dim[0] + dim[1]:
                step[j] = (fish[j] - barycenter[j]) * np.random.uniform(0, 1)
            elif j < dim[0] + dim[1] + dim[2]:
                step[j] = (fish[j] - barycenter[j]) * parameter_curr_step_volitive[0] * np.random.uniform(0, 1)
            elif j < dim[0] + dim[1] + dim[2] + dim[3]:
                step[j] = (fish[j] - barycenter[j]) * parameter_curr_step_volitive[1] * np.random.uniform(0, 1)
            elif j < dim[0] + dim[1] + dim[2] + dim[3] + dim[4]:
                step[j] = (fish[j] - barycenter[j]) * parameter_curr_step_volitive[2] * np.random.uniform(0, 1)
            elif j < dim[0] + dim[1] + dim[2] + dim[3] + dim[4] + dim[5]:
                step[j] = (fish[j] - barycenter[j]) * parameter_curr_step_volitive[3] * np.random.uniform(0, 1)
            elif model[1] == 'mixed' and j < total_dim:
                step[j] = (fish[j] - barycenter[j]) * parameter_curr_step_volitive[4] * np.random.uniform(0, 1)
        if all(q > 0 for q in step[0:dim[0]]):
            pos_threshold = max(step[0:dim[0]]) * thresh
            neg_threshold = 0
        elif all(q < 0 for q in step[0:dim[0]]):
            pos_threshold = 0
            neg_threshold = min(step[0:dim[0]]) * thresh
        else:
            pos_threshold = max(step[0:dim[0]]) * thresh
            neg_threshold = min(step[0:dim[0]]) * thresh
        for j in range(dim[0]):
            if step[j] > pos_threshold:
                step[j] = 1
            elif step[j] < neg_threshold:
                step[j] = -1
            else:
                step[j] = 0
        if dim[1] != 0:
            if all(q > 0 for q in step[dim[0]:dim[0]+dim[1]]):
                pos_bias = max(step[dim[0]:dim[0]+dim[1]]) * thresh
                neg_bias = 0
            elif all(q < 0 for q in step[dim[0]:dim[0]+dim[1]]):
                pos_bias = 0
                neg_bias = min(step[dim[0]:dim[0]+dim[1]]) * thresh
            else:
                pos_bias = max(step[dim[0]:dim[0]+dim[1]]) * thresh
                neg_bias = min(step[dim[0]:dim[0]+dim[1]]) * thresh
            for j in range(dim[0], dim[0]+dim[1]):
                if step[j] > pos_bias:
                    step[j] = 1
                elif step[j] < neg_bias:
                    step[j] = -1
                else:
                    step[j] = 0
        for j in range(total_dim):
            if curr_weight_school > prev_weight_school:
                new_position[j] = fish[j] - step[j]
            else:
                new_position[j] = fish[j] + step[j]
            if j < dim[0]:
                if new_position[j] < min_func:
                    new_position[j] = min_func
                elif new_position[j] > max_func:
                    new_position[j] = max_func
            elif j < dim[0] + dim[1] and dim[1] != 0:
                if new_position[j] < -1/2:
                    new_position[j] = -1/2 
                elif new_position[j] > 1/2:
                    new_position[j] = 1/2 
            elif j < dim[0] + dim[1] + dim[2]:
                if new_position[j] < search_area[0]:
                    new_position[j] = search_area[0] 
                elif new_position[j] > search_area[1]:
                    new_position[j] = search_area[1] 
            elif j < dim[0] + dim[1] + dim[2] + dim[3]:
                if new_position[j] < search_area[2]:
                    new_position[j] = search_area[2]
                elif new_position[j] > search_area[3]:
                    new_position[j] = search_area[3]
            elif j < dim[0] + dim[1] + dim[2] + dim[3] + dim[4]:
                if new_position[j] < search_area[4]:
                    new_position[j] = search_area[4]
                elif new_position[j] > search_area[5]:
                    new_position[j] = search_area[5]
            elif j < dim[0] + dim[1] + dim[2] + dim[3] + dim[4] + dim[5]:
                if new_position[j] < search_area[6]:
                    new_position[j] = search_area[6]
                elif new_position[j] > search_area[7]:
                    new_position[j] = search_area[7]
            elif model[1] == 'mixed' and j < total_dim:
                if new_position[j] < search_area[8]:
                    new_position[j] = search_area[8]
                elif new_position[j] > search_area[9]:
                    new_position[j] = search_area[9]
        school_update_volitive.append(new_position)
        if model[1] == 'mixed':
            cost_volitive, record = abcsmc(N, all_poss_state, model, observation, dim, num_gene, new_position, lam, num_sample, known_net, known_bias, M, tolerance)
            #cost_volitive = abcsmc(N, all_poss_state, model, observation, dim, num_gene, new_position, lam, num_sample, known_net, known_bias, M, tolerance)
        else:
            cost_volitive = apf(N, all_poss_state, model, observation, dim, num_gene, new_position, lam, num_sample, known_net, known_bias)
        cost.append(cost_volitive)
        if cost_volitive > best_cost:
            best_cost = cost_volitive
            best_fish = new_position
    if model[1] == 'mixed':
        return curr_weight_school, school_update_volitive, best_fish, cost, best_cost, record
    else:
        return curr_weight_school, school_update_volitive, best_fish, cost, best_cost

def update_step(total_iter, curr_iter, parameter_step_individual_init, parameter_step_individual_final):
    '''
    Input:
            total_iter: the number of total iteration
            curr_iter: the number of current iteration
    Output:
            updated step size -> linearly reduced after each iteration
    '''
    parameter_curr_step_individual = parameter_step_individual_init - curr_iter * (parameter_step_individual_init - parameter_step_individual_final) / total_iter
    parameter_curr_step_volitive = 2 * parameter_curr_step_individual
    return parameter_curr_step_individual, parameter_curr_step_volitive

def sample(school_size , min_func, max_func, search_area, dim, model):
    '''
    sample the position of fish uniformly in the search space
    '''
    total_dim = sum(dim)
    if model[1] == 'mixed':
        net = np.zeros((school_size, dim[0]))
        bias = np.zeros((school_size, dim[1]))
        noise = np.zeros((school_size, dim[2]))
        shape = np.zeros((school_size, dim[3]))
        scale = np.zeros((school_size, dim[4]))
        vari = np.zeros((school_size, dim[5]))
        fold = np.zeros((school_size, dim[6]))
        school = np.zeros((school_size, total_dim))
        for i in range(school_size):
            net[i] = np.random.uniform(min_func, max_func, dim[0])
            bias[i] = np.random.uniform(min_func, max_func, dim[1])
            noise[i] = np.random.uniform(search_area[0], search_area[1], dim[2])
            shape[i] = np.random.uniform(search_area[2], search_area[3], dim[3])
            scale[i] = np.random.uniform(search_area[4], search_area[5], dim[4])
            vari[i] = np.random.uniform(search_area[6], search_area[7], dim[5])
            fold[i] = np.random.uniform(search_area[8], search_area[9], dim[6])
            school[i] = np.append(np.append(np.append(np.append(np.append(np.append(np.rint(net[i]), np.array([1/2 if i > 0 else -1/2 for i in bias[i]])), noise[i]), shape[i]), scale[i]), vari[i]), fold[i])
    else:
        net = np.zeros((school_size, dim[0]))
        bias = np.zeros((school_size, dim[1]))
        noise = np.zeros((school_size, dim[2]))
        base = np.zeros((school_size, dim[3]))
        delta = np.zeros((school_size, dim[4]))
        vari = np.zeros((school_size, dim[5]))
        school = np.zeros((school_size, total_dim))
        for i in range(school_size):
            net[i] = np.random.uniform(min_func, max_func, dim[0])
            bias[i] = np.random.uniform(min_func, max_func, dim[1])
            noise[i] = np.random.uniform(search_area[0], search_area[1], dim[2])
            base[i] = np.random.uniform(search_area[2], search_area[3], dim[3])
            delta[i] = np.random.uniform(search_area[4], search_area[5], dim[4])
            vari[i] = np.random.uniform(search_area[6], search_area[7], dim[5])
            school[i] = np.append(np.append(np.append(np.append(np.append(np.rint(net[i]), np.array([1/2 if i > 0 else -1/2 for i in bias[i]])), noise[i]), base[i]), delta[i]), vari[i])
    return school 

def fish_school_search(dim, num_gene, model, observation, all_poss_state, school_size, num_iterations, N, lam, search_area, num_sample, known_net, known_bias, M, tolerance, min_func= -1, max_func= 1, min_weight = 1):
    '''
    fish school search algorithm:
    1. individual movement
    2. feeding
    3. collective instinctive movement
    4. collective volitive movement
    '''
    total_dim = sum(dim)
    max_weight = num_iterations / 2
    best_cost = - np.inf
    best_fish = np.zeros((total_dim,))
    order = np.arange(int(search_area.shape[0] / 2))
    parameter_step_individual_init = 0.1 * (search_area[2 * order + 1] - search_area[2 * order])
    parameter_step_individual_final = 0.0001 * parameter_step_individual_init
    parameter_curr_step_individual = parameter_step_individual_init
    parameter_curr_step_volitive = parameter_curr_step_individual * 2
    curr_weight_school = 0.0
    cost = []
    school = sample(school_size, min_func, max_func, search_area, dim, model)
    for i in range(school_size):
        position = school[i]
        if model[1] == 'mixed':
            likelihood, record = abcsmc(N, all_poss_state, model, observation, dim, num_gene, position, lam, num_sample, known_net, known_bias, M, tolerance)
            #likelihood = abcsmc(N, all_poss_state, model, observation, dim, num_gene, position, lam, num_sample, known_net, known_bias, M, tolerance)
        else:
            likelihood = apf(N, all_poss_state, model, observation, dim, num_gene, position, lam, num_sample, known_net, known_bias)
        cost.append(likelihood)
    weight = [(max_weight / 2.0) for _ in range(len(school))]
    for j in tqdm(range(num_iterations)):
        thresh = j/num_iterations
        if model[1] == 'mixed':
            [best_fish, best_cost, school_update_individual, cost, delta_position_update_individual, delta_cost_update_individual, record] = individual_movement(best_fish, best_cost, cost, school, parameter_curr_step_individual, max_func, min_func, search_area, dim, num_gene, model, observation, thresh, all_poss_state, N, lam, num_sample, known_net, known_bias, M, tolerance)
        else:
            [best_fish, best_cost, school_update_individual, cost, delta_position_update_individual, delta_cost_update_individual] = individual_movement(best_fish, best_cost, cost, school, parameter_curr_step_individual, max_func, min_func, search_area, dim, num_gene, model, observation, thresh, all_poss_state, N, lam, num_sample, known_net, known_bias, M, tolerance)
        weight = feeding(weight, max_weight, min_weight, delta_cost_update_individual)
        school_update_instinctive = collective_instinctive_movement(school_update_individual, dim, delta_position_update_individual, delta_cost_update_individual, min_func, max_func, search_area, thresh, model)
        if model[1] == 'mixed':
            [curr_weight_school, school, best_fish, cost, best_cost, record] = collective_volitive_movement(best_fish, best_cost, curr_weight_school, weight, dim, school_update_instinctive, min_func, max_func, search_area, parameter_curr_step_volitive, num_gene, model, observation, thresh, all_poss_state, N, lam, num_sample, known_net, known_bias, M, tolerance)
        else:
            [curr_weight_school, school, best_fish, cost, best_cost] = collective_volitive_movement(best_fish, best_cost, curr_weight_school, weight, dim, school_update_instinctive, min_func, max_func, search_area, parameter_curr_step_volitive, num_gene, model, observation, thresh, all_poss_state, N, lam, num_sample, known_net, known_bias, M, tolerance)
        [parameter_curr_step_individual, parameter_curr_step_volitive] = update_step(num_iterations, j, parameter_step_individual_init, parameter_step_individual_final)
    if model[1] == 'mixed':
        return best_cost, best_fish, school, record
    else:
        return best_cost, best_fish, school

