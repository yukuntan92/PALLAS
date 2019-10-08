#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from apf import apf
from tqdm import tqdm

def individual_movement(best_fish, best_cost, cost, school, curr_step_individual, parameter_curr_step_individual, max_func, min_func, search_area, dim, num_gene, bias, model, observation, thresh, A, N, lam, num_sample):
    total_dim = sum(dim)
    school_update_individual = []
    delta_cost_update_individual = []
    delta_position_update_individual = []
    for k in range(len(school)):
        new_position = np.zeros((total_dim,), dtype=np.float)
        random_step = np.zeros((total_dim,), dtype=np.float)
        for i in range(total_dim):
            if i < dim[0]:
                random_step[i] = curr_step_individual * np.random.uniform(-1, 1)
            if i < dim[0] + dim[1]:
                random_step[i] = parameter_curr_step_individual[0] * np.random.uniform(-1, 1)
            if i < dim[0] + dim[1] + dim[2]:
                random_step[i] = parameter_curr_step_individual[1] * np.random.uniform(-1, 1)
            if i < dim[0] + dim[1] + dim[2] + dim[3]:
                random_step[i] = parameter_curr_step_individual[2] * np.random.uniform(-1, 1)
            if i < total_dim:
                random_step[i] = parameter_curr_step_individual[3] * np.random.uniform(-1, 1)
        if [ q > 0 for q in random_step[0:dim[0]]] == [True] * dim[0]:
            pos_threshold = max(random_step[0:dim[0]]) * thresh
            neg_threshold = 0
        elif [ q < 0 for q in random_step[0:dim[0]]] == [True] * dim[0]:
            pos_threshold = 0
            neg_threshold = min(random_step[0:dim[0]]) * thresh
        else:
            pos_threshold = max(random_step[0:dim[0]]) * thresh
            neg_threshold = min(random_step[0:dim[0]]) * thresh
        for i in range(dim[0]):
            if random_step[i] > pos_threshold:
                random_step[i] = 1
            elif random_step[i] < neg_threshold:
                random_step[i] = -1
            else:
                random_step[i] = 0
        for i in range(total_dim):
            new_position[i] = school[k][i] + random_step[i]
            if i < dim[0]:
                if new_position[i] < min_func:
                    new_position[i] = min_func
                elif new_position[i] > max_func:
                    new_position[i] = max_func
            elif i < dim[0]+dim[1]:
                if new_position[i] < search_area[0]:
                    new_position[i] = search_area[0] 
                elif new_position[i] > search_area[1]:
                    new_position[i] = search_area[1] 
            elif i < dim[0]+dim[1]+dim[2]:
                if new_position[i] < search_area[2]:
                    new_position[i] = search_area[2]
                elif new_position[i] > search_area[3]:
                    new_position[i] = search_area[3]
            elif i < dim[0]+dim[1]+dim[2]+dim[3]:
                if new_position[i] < search_area[4]:
                    new_position[i] = search_area[4]
                elif new_position[i] > search_area[5]:
                    new_position[i] = search_area[5]
            else:
                if new_position[i] < search_area[6]:
                    new_position[i] = search_area[6]
                elif new_position[i] > search_area[7]:
                    new_position[i] = search_area[7]
#        for i in range(dim[2]):
#            if new_position[dim[0]+i] < 0:
#                new_position[dim[0]+i] = 0
        curr_cost = apf(N, A, model, observation, dim, num_gene, new_position, bias, lam, num_sample)
        if curr_cost > cost[k]:
            delta_cost = abs(curr_cost - cost[k])
            cost[k] = curr_cost
            delta_position = np.zeros((total_dim,), dtype=np.float)
            delta_position = new_position - school[k]
            school_update_individual.append(new_position) 
            if curr_cost > best_cost:
                best_cost = curr_cost
                best_fish = new_position
        else:
            delta_position = np.zeros((total_dim,), dtype=np.float)
            delta_cost = 0
            school_update_individual.append(school[k])
        delta_cost_update_individual.append(delta_cost)
        delta_position_update_individual.append(delta_position)
    return best_fish, best_cost, school_update_individual, cost, delta_position_update_individual, delta_cost_update_individual

def max_delta_cost(delta_cost):
    max_delta_cost = 0
    for cost in delta_cost:
        if max_delta_cost < cost:
            max_delta_cost = cost
    return max_delta_cost

def feeding(weight, max_weight, min_weight, delta_cost):
    max_delta = max_delta_cost(delta_cost)
    for i in range(len(delta_cost)):
        if max_delta:
            weight[i] += (delta_cost[i] / max_delta)
        if weight[i] > max_weight:
            weight[i] = max_weight
        elif weight[i] < min_weight:
            weight[i] = min_weight
    return weight

def collective_instinctive_movement(school, dim, delta_position, delta_cost, min_func, max_func, search_area, thresh):
    total_dim = sum(dim)
    individual_move_cost = np.zeros((total_dim,), dtype=np.float)
    density = 0.0
    school_update_instinctive = []
    for i in range(len(school)):
        density += delta_cost[i]
        for k in range(total_dim):
            individual_move_cost[k] += (delta_position[i][k] * delta_cost[i])
    for j in range(total_dim):
        if density != 0:
            individual_move_cost[j] = individual_move_cost[j] / density
    if [ q > 0 for q in individual_move_cost[0:dim[0]]] == [True] * dim[0]:
        pos_threshold = max(individual_move_cost[0:dim[0]]) * thresh
        neg_threshold = 0
    elif [ q < 0 for q in individual_move_cost[0:dim[0]]] == [True] * dim[0]:
        pos_threshold = 0
        neg_threshold = min(individual_move_cost[0:dim[0]]) * thresh
    else:
        pos_threshold = max(individual_move_cost[0:dim[0]]) * thresh
        neg_threshold = min(individual_move_cost[0:dim[0]]) * thresh
    for k in range(dim[0]):
        if individual_move_cost[k] > pos_threshold:
            individual_move_cost[k] = 1
        elif individual_move_cost[k] < neg_threshold:
            individual_move_cost[k] = -1
        else:
            individual_move_cost[k] = 0
    for position in school:
        new_position = np.zeros((total_dim,), dtype=np.float)
        for k in range(total_dim):
            new_position[k] = position[k] + individual_move_cost[k]
            if k < dim[0]:
                if new_position[k] < min_func:
                    new_position[k] = min_func
                elif new_position[k] > max_func:
                    new_position[k] = max_func
            elif k < dim[0]+dim[1]:
                if new_position[k] < search_area[0]:
                    new_position[k] = search_area[0] 
                elif new_position[k] > search_area[1]:
                    new_position[k] = search_area[1] 
            elif k < dim[0]+dim[1]+dim[2]:
                if new_position[k] < search_area[2]:
                    new_position[k] = search_area[2]
                elif new_position[k] > search_area[3]:
                    new_position[k] = search_area[3]
            elif k < dim[0]+dim[1]+dim[2]+dim[3]:
                if new_position[k] < search_area[4]:
                    new_position[k] = search_area[4]
                elif new_position[k] > search_area[5]:
                    new_position[k] = search_area[5]
            else:
                if new_position[k] < search_area[6]:
                    new_position[k] = search_area[6]
                elif new_position[k] > search_area[7]:
                    new_position[k] = search_area[7]
#        for i in range(dim[2]):
#            if new_position[dim[0]+i] < 0:
#                new_position[dim[0]+i] = 0
        school_update_instinctive.append(new_position)
    return school_update_instinctive

def calculate_weight(curr_weight_school, weight):
    prev_weight_school = curr_weight_school
    curr_weight_school = 0.0
    for i in range(len(weight)):
        curr_weight_school += weight[i]
    return curr_weight_school, prev_weight_school

def calculate_barycenter(weight, dim, school):
    total_dim = sum(dim)
    barycenter = np.zeros((total_dim,), dtype=np.float)
    density = 0.0
    for i in range(len(school)):
        density += weight[i]
        for j in range(total_dim):
            barycenter[j] += (school[i][j] * weight[i])
    for k in range(total_dim):
        barycenter[k] = barycenter[k] / density
    return barycenter

def collective_volitive_movement(best_fish, best_cost, curr_weight_school, weight, dim, school, min_func, max_func, search_area, curr_step_volitive, parameter_curr_step_volitive, num_gene, bias, model, observation, thresh, A, N, lam, num_sample):
    total_dim = sum(dim)
    [curr_weight_school, prev_weight_school] = calculate_weight(curr_weight_school, weight)
    barycenter = calculate_barycenter(weight, dim, school)
    school_update_volitive = []
    cost = []
    for j in range(len(school)):
        new_position = np.zeros((total_dim,), dtype=np.float)
        step = np.zeros((total_dim,), dtype=np.float)
        for i in range(total_dim):
            if i < dim[0]:
                step[i] = (school[j][i] - barycenter[i]) * curr_step_volitive * np.random.uniform(0,1)
            if i < dim[0] + dim[1]:
                step[i] = (school[j][i] - barycenter[i]) * parameter_curr_step_volitive[0] * np.random.uniform(0,1)
            if i < dim[0] + dim[1] + dim[2]:
                step[i] = (school[j][i] - barycenter[i]) * parameter_curr_step_volitive[1] * np.random.uniform(0,1)
            if i < dim[0] + dim[1] + dim[2] + dim[3]:
                step[i] = (school[j][i] - barycenter[i]) * parameter_curr_step_volitive[2] * np.random.uniform(0,1)
            if i < total_dim:
                step[i] = (school[j][i] - barycenter[i]) * parameter_curr_step_volitive[3] * np.random.uniform(0,1)
        if [ q > 0 for q in step[0:dim[0]]] == [True] * dim[0]:
            pos_threshold = max(step[0:dim[0]]) * thresh
            neg_threshold = 0
        elif [ q < 0 for q in step[0:dim[0]]] == [True] * dim[0]:
            pos_threshold = 0
            neg_threshold = min(step[0:dim[0]]) * thresh
        else:
            pos_threshold = max(step[0:dim[0]]) * thresh
            neg_threshold = min(step[0:dim[0]]) * thresh
        for i in range(dim[0]):
            if step[i] > pos_threshold:
                step[i] = 1
            elif step[i] < neg_threshold:
                step[i] = -1
            else:
                step[i] = 0
        for i in range(total_dim):
            if curr_weight_school > prev_weight_school:
                new_position[i] = school[j][i] - step[i]
            else:
                new_position[i] = school[j][i] + step[i]
            if i < dim[0]:
                if new_position[i] < min_func:
                    new_position[i] = min_func
                elif new_position[i] > max_func:
                    new_position[i] = max_func
            elif i < dim[0]+dim[1]:
                if new_position[i] < search_area[0]:
                    new_position[i] = search_area[0] 
                elif new_position[i] > search_area[1]:
                    new_position[i] = search_area[1] 
            elif i < dim[0]+dim[1]+dim[2]:
                if new_position[i] < search_area[2]:
                    new_position[i] = search_area[2]
                elif new_position[i] > search_area[3]:
                    new_position[i] = search_area[3]
            elif i < dim[0]+dim[1]+dim[2]+dim[3]:
                if new_position[i] < search_area[4]:
                    new_position[i] = search_area[4]
                elif new_position[i] > search_area[5]:
                    new_position[i] = search_area[5]
            else:
                if new_position[i] < search_area[6]:
                    new_position[i] = search_area[6]
                elif new_position[i] > search_area[7]:
                    new_position[i] = search_area[7]
#        for i in range(dim[2]):
#            if new_position[dim[0]+i] < 0:
#                new_position[dim[0]+i] = 0
        school_update_volitive.append(new_position)
        cost_volitive = apf(N, A, model, observation, dim, num_gene, new_position, bias, lam, num_sample)
        cost.append(cost_volitive)
        if cost_volitive > best_cost:
            best_cost = cost_volitive
            best_fish = new_position
    return curr_weight_school, school_update_volitive, best_fish, cost, best_cost

def update_step(total_iter, curr_iter, step_individual_init, step_individual_final, parameter_step_individual_init, parameter_step_individual_final):
    curr_step_individual = step_individual_init - curr_iter * float(step_individual_init - step_individual_final) / total_iter
    curr_step_volitive = 2 * curr_step_individual
    parameter_curr_step_individual = parameter_step_individual_init - curr_iter * (parameter_step_individual_init - parameter_step_individual_final) / total_iter
    parameter_curr_step_volitive = 2 * parameter_curr_step_individual
    return curr_step_individual, curr_step_volitive, parameter_curr_step_individual, parameter_curr_step_volitive

def sample(school_size , min_func, max_func, search_area, dim):
    total_dim = sum(dim)
    x = np.zeros((school_size, dim[0]))
    noise = np.zeros((school_size, dim[1]))
    base = np.zeros((school_size, dim[2]))
    delta = np.zeros((school_size, dim[3]))
    disp = np.zeros((school_size, dim[4]))
    y = np.zeros((school_size, total_dim))
    for i in range(school_size):
        x[i] = np.random.uniform(min_func, max_func, dim[0])
        noise[i] = np.random.uniform(search_area[0], search_area[1], dim[1])
        base[i] = np.random.uniform(search_area[2], search_area[3], dim[2])
        delta[i] = np.random.uniform(search_area[4], search_area[5], dim[3])
        disp[i] = np.random.uniform(search_area[6], search_area[7], dim[4])
        y[i] = np.append(np.append(np.append(np.append(np.rint(x[i]),noise[i]), base[i]), delta[i]), disp[i])
    return y

def fish_school_search(dim, num_gene, bias, model, observation, A, school_size, num_iterations, N, lam, search_area, num_sample, min_func=-1, max_func=1, step_individual_init = 0.2, step_individual_final = 0.0002, step_volitive_init = 0.02, step_volitive_final = 0.002, min_weight = 1):
    it = np.arange(int(search_area.shape[0]/2))
    parameter_step_individual_init = 0.1* (search_area[2*it+1] - search_area[2*it])
    parameter_step_individual_final = parameter_step_individual_init/10000
    total_dim = sum(dim)
    max_weight = num_iterations/2
    best_cost = - np.inf
    best_fish = np.zeros((total_dim,), dtype=np.float)
    curr_step_individual = step_individual_init
    curr_step_volitive = curr_step_individual * 2
    parameter_curr_step_individual = parameter_step_individual_init
    parameter_curr_step_volitive = parameter_curr_step_individual * 2
    curr_weight_school = 0.0
    cost = []
    school = sample(school_size, min_func, max_func, search_area, dim)
    for idx in range(school_size):
        position = school[idx]
        costs = apf(N, A, model, observation, dim, num_gene, position, bias, lam, num_sample)
        cost.append(costs)
    weight = [(max_weight / 2.0) for _ in range(len(school))]
    for j in tqdm(range(num_iterations)):
        thresh = j/num_iterations
        [best_fish, best_cost, school_update_individual, cost, delta_position_update_individual, delta_cost_update_individual] = individual_movement(best_fish, best_cost, cost, school, curr_step_individual, parameter_curr_step_individual, max_func, min_func, search_area, dim, num_gene, bias, model, observation, thresh, A, N, lam, num_sample)
        weight = feeding(weight, max_weight, min_weight, delta_cost_update_individual)
        school_update_instinctive = collective_instinctive_movement(school_update_individual, dim, delta_position_update_individual, delta_cost_update_individual, min_func, max_func, search_area, thresh)
        [curr_weight_school, school, best_fish, cost, best_cost] = collective_volitive_movement(best_fish, best_cost, curr_weight_school, weight, dim, school_update_instinctive, min_func, max_func, search_area, curr_step_volitive, parameter_curr_step_volitive, num_gene, bias, model, observation, thresh, A, N, lam, num_sample)
        [curr_step_individual, curr_step_volitive, parameter_curr_step_individual, parameter_curr_step_volitive] = update_step(num_iterations, j, step_individual_init, step_individual_final, parameter_step_individual_init, parameter_step_individual_final)
#        print(cost)
        #print(school)
        #print(best_cost)
        #print(best_fish)
     #print(best_cost/sum(cost))
     #print(best_fish)
     #print(school)
     #print(np.rint(best_fish))
    return best_cost, best_fish, school

