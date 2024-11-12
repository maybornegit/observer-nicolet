#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 13:51:08 2024

@author: morganmayborne
"""
import csv
import matplotlib.pyplot as plt
import numpy as np

file_name = '/Users/morganmayborne/Downloads/traj_ground.csv'
in_conv = 0.0254

traj_list = []
with open(file_name, 'r') as file:
    reader = csv.reader(file)
    read_list = list(reader)
    for i in range(len(read_list)//2):
        traj_list.append([])
        traj_list[-1].append(read_list[2*i])
        traj_list[-1].append(read_list[2*i+1])
 
new_trajs = []
factors = [2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5]
cutoffs = [2,5,8,11,16,22,26,30]
factor = 0
for i in range(len(traj_list)):
    cur_traj = np.array(traj_list[i],dtype=np.float64)
    for j in range(cur_traj.shape[1]):
        for k in range(len(factors)):
            if cur_traj[1,j] <= cutoffs[k]:
                cur_traj[1,j] /= np.pi*((factors[k]/2*in_conv)**2)*1000
                break
            if k == len(factors)-1:
                cur_traj[1,j] /= np.pi*((factors[-1]/2*in_conv)**2)*1000
        cur_traj[0,j] += 7
        
    new_trajs.append(cur_traj)

plt.figure(0)
# for i in range(len(traj_list)):
i = 10
plt.plot(np.array(new_trajs[i][0],dtype=np.float64),np.array(new_trajs[i][1],dtype=np.float64),label='M_fm')
plt.title("Example Trajectory")
plt.xlabel("Time (Days)")
plt.ylabel("Fresh Biomass (kg m-2)")
plt.legend()
plt.grid(True)
plt.show()
        
