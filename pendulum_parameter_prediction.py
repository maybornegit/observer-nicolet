#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 09:31:10 2024

@author: morganmayborne
"""

import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from scipy.signal import place_poles

plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pacoty.mplstyle')

### Pendulum Model

g = 9.81
l = 11
dt = 0.001  # seconds
total_time = 100
t = np.arange(0,total_time,dt)
th_result = (np.pi/48)*np.cos(t*np.sqrt(g/l))
x = l*np.sin(th_result)
y = l*np.cos(th_result)

# SP ~ dd theta = -(g/l)*sin(theta)

# plt.plot(t, th_result, label='Theta Result')
# plt.title('Simple Pendulum Oscillation')
# plt.legend()
# plt.xlabel('Time (seconds)')
# plt.ylabel('Theta Value')
# plt.grid(True)
# plt.show()


###############################

# Observer Below Can Find the Length of the Pendulum to a Moderate Degree

steps = len(t)
scale = 10
scale_im = 1j
pole_placement_iter = 1
initial_guess = np.pi/48
C = np.array([[1,0,0]])
guesses = np.zeros((steps, 3))
param_guess = np.zeros((steps, 1))
guesses[0,:] = np.array([initial_guess, 0, 11.5])
param_guess[0,0] = guesses[0,2]

A = np.array([[0,1,0],[-g/guesses[0,2], 0,g*guesses[0,0]/guesses[0,2]],[0,0,0]])
K_o = place_poles(A.T, C.T, np.array([-scale,-scale+scale_im,-scale-scale_im])).gain_matrix.T

for i in range(1,steps):
    if i % pole_placement_iter == 0 and np.abs(guesses[i-1,0]) >= 1e-3: 
        A = np.array([[0,1,0],[-g/guesses[i-1,2], 0,g*guesses[i-1,0]/guesses[i-1,2]],[0,0,0]])
        eigvals_system = np.linalg.eigvals(A)
        # print(eigvals_system)
        K_o = place_poles(A.T, C.T,np.array([-scale,-scale+scale_im,-scale-scale_im])).gain_matrix.T
    
    guess_dot = np.array([guesses[i-1,1], -(g/guesses[i-1,2])*np.sin(guesses[i-1, 0]),0])+K_o@np.array([th_result[i]-guesses[i-1, 0]])
    guesses[i,:] = guess_dot*dt+guesses[i-1,:]
    guesses[i,2] = max(0.1, guesses[i,2])
    param_guess[i,0] = np.mean(guesses[:(i+1),2])
    if i % 100 == 0:
        print('i',i,'l:',param_guess[i,0])
    

plt.plot(t, th_result,'r',label='Theta Result')
plt.plot(t, guesses[:,0],'b--',label='Observer Result')
plt.title('Theta State Prediction')
plt.legend(facecolor='white',frameon=True, loc='upper right')
plt.xlabel('Time (seconds)')
plt.ylabel('Theta Value (in rad)')
plt.grid(True)
plt.show()  

plt.plot(t, guesses[:,2],label='Instantaneous Parameter Guess')
plt.plot(t, param_guess[:,0], '--',label='Averaged Parameter Guess')
plt.plot(t, 11*np.ones(guesses[:,2].shape), label='True Value')
plt.title('Length Parameter Estimation')
plt.legend()
plt.xlabel('Time (seconds)')
plt.ylabel('Length (in m)')
plt.grid(True)
plt.show()  
    
print()
    
    





