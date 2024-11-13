#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 13:51:08 2024

@author: morganmayborne
"""
import csv
import matplotlib.pyplot as plt
import numpy as np
from nicolet_model_base import basic_test_w_lighting, residual_function_k, residual_function_va
from scipy.optimize import minimize

file_name = '/Users/morganmayborne/Downloads/traj_ground.csv'
in_conv = 0.0254
k_param = 0.4e-6
v = 22.1
a = .5
bounds_k = [(0.375e-6, .425e-6)]
bounds_va = [(v*.25, v*5), (0.3, 0.7)]

##### Read trajectories
traj_list = []
with open(file_name, 'r') as file:
    reader = csv.reader(file)
    read_list = list(reader)
    for i in range(len(read_list)//2):
        traj_list.append([])
        traj_list[-1].append(read_list[2*i])
        traj_list[-1].append(read_list[2*i+1])
 
##### Find trajectories divided by area
new_trajs = []
factors = [2.0,2.5,3.0,3.5,4.0,4.5,5.0]
cutoffs = [2,5,8,11,16,22,26]

spacing = 0.02
factor = 40*46*(in_conv)**2/(((40*in_conv-2*spacing)/spacing)*((46*in_conv-2*spacing)/spacing))
factor = 1/1000
for i in range(len(traj_list)):
    cur_traj = np.array(traj_list[i],dtype=np.float64)
    for j in range(cur_traj.shape[1]):
        for k in range(len(factors)):
            if cur_traj[1,j] <= cutoffs[k]:
                cur_traj[1,j] /= (factors[k]*4.5*in_conv**2)*1000
                # cur_traj[1,j] /= factor*1000
                break
            if k == len(factors)-1:
                # cur_traj[1,j] /= factor*1000
                cur_traj[1,j] /= (factors[-1]*4.5*in_conv**2)*1000
        cur_traj[0,j] += -4
        
    new_trajs.append(cur_traj)

##### Light Test for Test Trajectories
light_test = [293,357,271.1,225,265,293,318,328,356,318,302,339,228,356,367]
i_s = [0,1,2,3,10,11,12,13,25,26,27,41,42,43,44]
result_k = minimize(residual_function_k, np.array([k_param]),args=([new_trajs,light_test,4,v,a,i_s],), method='Nelder-Mead', bounds=bounds_k)
print('Minimized Parameters: ',result_k.x)  # Optimal values
print('Minimized Function: ',result_k.fun)  # Minimum function value

residual = 0
for j, i in enumerate(i_s):
    plt.figure(j)
    plt.plot(np.array(new_trajs[i][0],dtype=np.float64),np.array(new_trajs[i][1],dtype=np.float64),label='M_fm - Ground Truth')
    # plt.title("Example Trajectory")
    # plt.xlabel("Time (Days)")
    # plt.ylabel("Fresh Biomass (kg m-2)")
    # plt.legend()
    # plt.grid(True)

    lighting = light_test[i_s.index(i)]
    # lightings = [15,20,25,30,35,40,45]
    # for k, lighting in enumerate(lightings):
    # t_traj, x_traj, y_traj = basic_test_w_lighting(lighting, k_param, v, a)

    # lim = []
    # corr = []
    # for k,t in enumerate(t_traj):
    #     if t/60/60/24 >= np.min(new_trajs[i][0]) and len(lim) == 0:
    #         lim.append(k)
    #     elif t/60/60/24 >= np.max(new_trajs[i][0]) and len(lim) == 1:
    #         lim.append(k)
    #     if t/60/60/24 in new_trajs[i][0]:
    #         corr.append(k)

    # resid = 0
    # for k in range(len(new_trajs[i][0])):
    #     resid += (new_trajs[i][1][k] - y_traj[1,corr[k]])**2
    # print("Residual: ",resid)

    # Nelder-Mead optimization
    result = minimize(residual_function_va, np.array([v,a]),args=([new_trajs[i],lighting,4,result_k.x[0]],), method='Nelder-Mead', bounds=bounds_va)

    print('Minimized Parameters: ',result.x)  # Optimal values
    print('Minimized Function: ',result.fun)  # Minimum function value
    residual += result.fun

    t_traj, x_traj, y_traj = basic_test_w_lighting(lighting, result_k.x[0], result.x[0], result.x[1], 1)

    lim = []
    corr = []
    for k,t in enumerate(t_traj):
        if t/60/60/24 >= np.min(new_trajs[i][0]) and len(lim) == 0:
            lim.append(k)
        elif t/60/60/24 >= np.max(new_trajs[i][0]) and len(lim) == 1:
            lim.append(k)
        if t/60/60/24 in new_trajs[i][0]:
            corr.append(k)

    plt.plot(t_traj[lim[0]:lim[1]]/60/60/24, y_traj[1,lim[0]:lim[1]].T, label="M_fm - Simulation")
    plt.title("Simulation Run")
    plt.xlabel('Time (days)')
    plt.ylabel('Weight (kg m-2)')
    plt.legend()
    plt.grid(True)

print(residual)
plt.show()
    



