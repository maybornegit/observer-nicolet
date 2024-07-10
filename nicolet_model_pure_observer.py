#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 16:19:13 2024

@author: morganmayborne
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from scipy.signal import place_poles
import scipy.signal as signal
from scipy.ndimage import gaussian_filter1d
from scipy.fft import fft, fftfreq

from nicolet_model_base import * 

plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pacoty.mplstyle')

####### Parameters for the Nicolet Model ##########

eps = 0.055            #0 - apparent light use efficiency
sigma = 1.4e-3         #1 - co2 transport coefficient
co2_baseline = 0.0011  #2 - co2 compensation point
a = .5                 #3 - leaf area closure paramete
b_p = 0.8              #4 - threshold paramter of photosynthesis inhibition function
pi_v = 580             #5 - osmotic pressure in the vacuoles
lambda_ = 1/1200       #6 - carbon concentration calculation parameter
gamma = 0.61           #7 - coefficient of osmotic carbon equivalence
s_p = 10               #8 - slope parameter of photosynthesis inhibition function
K = 0.4e-6             #9 - maintenance respiration coefficient
c = 0.0693             #10 - temp effect paramter
t_baseline = 20        #11 - reference temp.
theta = 0.3            #12 - growth respiration loss factor
v = 22.1*.5            #13 - growth rate coefficient without inhibition from closed canopy
b_g = 0.2              #14 - threshold paramter of growth inhibition function
s_g = 10               #15 - slope parameter of growth inhibition function
eta_OMC = 0.03         #16 - organic matter in kg per mol C
eta_MMN = 0.148        #17 - minerals in kg per mol N in vacuoles
beta = 6.0             #18 - regression paramter of C/N ratio in vacuoles
eta_NO3N = 0.062       #19 - kg nitrate per mol N

## Parameter Array (see above for indices) ##
p = np.array([eps, sigma, co2_baseline, a, b_p, pi_v, lambda_, gamma, s_p, K, c, t_baseline, theta, v, b_g, s_g, eta_OMC, eta_MMN, beta, eta_NO3N])

## Timespan Set-up ##
day_max = 200 # Length of Model Projection (in days)
t_span = (0, 86400*day_max)  # Time span
dt = 3600  # Time steps for model (in seconds)
t_array = np.arange(t_span[0], t_span[1], dt)

## Initial Conditions for State / Input ##
x0 = np.array([0.007, 0.0671])  # Initial condition (M_cv, M_cs)
u = np.array([175,28.0,450])  # Input (I [W/m2],T[C],C_co2[ppm])
u *= np.array([2.1e-5,1,.0195/450]) # Conversion to Model Units

# ## Initialize Test Input Array ##
u = u*np.ones((t_span[1]//dt,3)) # Expanding Input to Fill Full Trajectory
u[:,0] *= np.array([(1 if i*3600 % 86400 < 36000 else 0.05) for i in range(t_span[1]//dt)]) # Input PAR - Step Function
u[:,1] += 2.0*np.cos(2*np.pi*(t_array-3600)/86400)  # Input Temp. - Sinusodial Daily Changes
u[:,2] += np.random.normal(0,1.5,t_span[1]//dt)*.22*u[:,2]  # Input Co2 - Gaussian Changes

### Run the NICOLET Model (Un-comment run_model for graphical representations)
run_model(f, h, x0, u, p, t_span, dt, 'Base Case', file_id='k_IC_test')
t_traj, x_traj, y_traj = solve_system(f, h, x0, u, p, t_span, dt) # Basic run for model, no graphs
y_traj_true = y_traj.copy()

####### Initialize Observer System ##########

def fill_A(x, u, p, A, iter_):
    '''
    Updating linearized state derivative matrix A, with updated state variables
    
    Parameters:
    x: State Variables [M_cv, M_cs]
    u: Input Parameters [I, T, C_co2]
    p: NICOLET Model Parameters
    A: Linearized Matrix for State-Space Derivative Function F
            
    Returns: 
    A: Updated Linearized Derivative Function F
    '''
    if u.shape[0] != 3:
        u = u[iter_, :]
        
    eps = p[0]
    sigma = p[1]
    co2_baseline = p[2]
    a = p[3]
    b_p = p[4]
    pi_v = p[5]
    lambda_ = p[6]
    gamma = p[7]
    s_p = p[8]
    c = p[10]
    t_baseline = p[11]
    theta = p[12]
    v = p[13]
    b_g = p[14]
    s_g = p[15]
    
    M_cv = x[0]
    M_cs = x[1]
    k = x[2]
    
    I = u[0]
    T = u[1]
    C_co2 = u[2]

    # Piece-by-Piece Updating of Matrix A, using State-Space Functions
    A[0,0] = p_(u,p)*f_(x,p)*dh_p_mcv(x, p)-dh_g_mcv(x, p)*M_cs*e_(x, u, p)-f_(x,p)*g_(x, u, p)*dh_g_mcv(x, p)*theta-g_(x, u, p)*f_(x, p)*dh_g_mcv(x, p)
    A[0,1] = p_(u, p)*a*f_(x, p)*h_p(x, p)+p_(u, p)*dh_p_mcs(x, p)*f_(x, p)-g_(x, u, p)*a*theta*f_(x,p)*h_g(x, p)-g_(x, u, p)*a*np.exp(-a*M_cs)*h_g(x, p)-dh_g_mcs(x, p)*M_cs*e_(x, u, p)-e_(x, u, p)*h_g(x, p)-g_(x, u, p)*g_(x, u, p)*dh_g_mcs(x, p)*(1+theta)
    A[0,2] = -M_cs*h_g(x, p)*e_(x, u, p)*(1/k)-(1+theta)*(1/k)*g_(x, u, p)*f_(x, p)*h_g(x, p)
    A[0,3] = F_cav(x, u, p) - h_g(x,p)*F_cm(x, u, p) - F_cg(x, u, p) - F_cvs(x, u, p)
    A[1,0] = dh_g_mcv(x, p)*M_cs*e_(x, u, p)+g_(x, u, p)*f_(x, p)*dh_g_mcv(x, p)
    A[1,1] = a*g_(x, u, p)*np.exp(-a*M_cs)*h_g(x, p)+e_(x, u, p)*M_cs*dh_g_mcs(x, p)-e_(x, u, p)*(1-h_g(x, p))+dh_g_mcs(x, p)*g_(x,u,p)*f_(x, p)
    A[1,2] = -M_cs*(1-h_g(x, p))*e_(x, u, p)*(1/k)+g_(x, u, p)*(1/k)*f_(x, p)*h_g(x, p)
    A[1,3] = F_cvs(x,u,p) - (1-h_g(x,p))*F_cm(x, u, p)
    eigs = np.linalg.eigvals(A)
    # print(eigs)
    return A, eigs

### Cut off for observer prediction (note - observer performs weird with corner / sharp change in state derivatives)
max_val = np.max(y_traj[1,:])
for i in range(y_traj.shape[1]):
    if y_traj[1,i] >= .9*max_val:
        end_point = i
        break

### Univerisalize cutoff for all trajectories
x_traj = x_traj[:,:end_point]
y_traj = y_traj[:,:end_point]
t_traj = t_traj[:end_point]

### Initialize Linearized Matrices (A for function f, C for function h)
A = np.zeros((4,4))
C = np.array([[eta_MMN-(eta_OMC*gamma/beta), eta_MMN+(eta_OMC*lambda_*pi_v/beta), 0, 0], [eta_MMN-(eta_OMC*gamma/beta), eta_MMN+1000*lambda_+eta_OMC*lambda_*pi_v/beta, 0, 0]])

### Initialize Core Hyper-parameters
step_diff = 0.99               # Difference between actual parameter and initial guess
scale = 9e-6                   # Real, Negative Part of Poles for Observer Placement
scale_im = 2.777777e-4j         # Imaginary Part of Poles for Observer Placement
true_value = p[9]               # Actual Value for the Parameter being Observed
skip_num = 1                    # Number of Iterations between State Estimation Updates
obs_skip_num = 1               # Number of Iterations between Observer Matrix Updates
dt = t_traj[skip_num]-t_traj[0] # Functional dt based on skip_num
stable = True


### Initialize Core Matrices
x_init = np.array([x_traj[:,0][0], x_traj[:,0][1], true_value*step_diff,1]) # State Estimation Matrix
A, eigs = fill_A(x_init, u, p, A,0)  # A Matrix based on Initial State
K_o = place_poles(A.T, C.T,np.array([-scale+scale_im,-scale+(scale_im/3),-scale-(scale_im/3),-scale-scale_im]), maxiter=100).gain_matrix.T  # Initial Observer Matrix
guesses = np.zeros((4, x_traj.shape[1]//skip_num)) # For Logging state/parameter estimations
t_shaved = np.zeros((1,x_traj.shape[1]//skip_num)) # For x-axis of resultant plots
guesses[:,0] = x_init.copy()  # initialize
parameter_guess = np.zeros((1,x_traj.shape[1]//skip_num))  # For Special predictions for important parameter
parameter_guess[0,0] = x_init[2] # initialize
stable_poles = np.zeros((1,x_traj.shape[1]//skip_num))


####### Core Observer Prediction Loop ##########

for i in range(1,x_traj.shape[1]):
    if i // skip_num == x_traj.shape[1] // skip_num:
        ## for making the matrices work for plotting
        break
    
    if i % obs_skip_num == 0:
        ## Observer Matrix Update
        
        A, eigs = fill_A(x_init, u, p, A, i)
        K_tmp = place_poles(A.T, C.T,np.array([-scale+scale_im,-scale+(scale_im/3),-scale-(scale_im/3),-scale-scale_im]), maxiter=30)
        if K_tmp.rtol <= 1e-3:
            ## threshold for strong observer
            K_o = K_tmp.gain_matrix.T
    
    if i % skip_num == 0:
        ## State Estimation Update
        
        # Initialize Actual Output from Model at this Time Step
        y = np.array([y_traj[:,i][0], y_traj[:,i][1]])
        
        # Guess Next State based on Prev. State + its Derivative
        p[9] = x_init[2] # Change Paramter Matrix with Current Guess
        guess_term = np.hstack((f(x_init[0:2],u,p,i), np.array([0,0]))) # Derivative at Prev. State Estimate
        x_guess_0 = runge_kutta_4(f, x_init[0:2], u, p, dt,i)  # Get Next State from RK-4 Estimation
        x_guess_0 = np.array([x_guess_0[0], x_guess_0[1], x_init[2], 1]) # Re-initialize quantities for 3rd+4th terms
        y_guess = h(x_guess_0[0:2], p)[0:2] # Use h() to find output prediction
        
        # Observer Term Estimation
        error = (y-y_guess) # Error between Actual + Estimated Output
        observer_term = K_o@error  # Observer Prediction Term
        x_guess_1_dot = observer_term
        x_guess_1 = dt*x_guess_1_dot + x_init  # Basic State Estimation from Derivative
        x_guess_1 /= x_guess_1[-1]  # Normalize based on Final Term
        x_guess_1[:2] = x_guess_0[:2] # Re-initialize quantities for 1st+2nd terms
        # x_guess_1[2] = sorted([0.2e-6, x_guess_1[2], 0.6e-6])[1]  # Optional Bounding in Rational Numbers
        
        # Printing Results
        dx = np.array(error, dtype=np.float64)  # Difference in Output
        # print('k:',i//skip_num,x_guess_1[2],'dx:', np.linalg.norm(dx), 'y_real:', y[1], 'y_guess:', -error[1]+y[1]) # Terminal Logging for Prediction, Error and Outputs
        if np.abs(x_guess_1[2]- x_init[2]) >= 1e-7:
            ## Stop estimation from making unstable changes
            x_guess_1[2] = x_init[2]
        x_init = x_guess_1.copy() # Re-initialize x_init for next estimation
        
        # Logging Results
        guesses[:,i//skip_num] = x_guess_1.copy()
        t_shaved[0,i//skip_num] = t_traj[i]
        parameter_guess[:,i//skip_num] = np.mean(guesses[2,:(i//skip_num)+1]) # Taking a Rolling Average for Better Estimation
        
        if eigs[0] <= -3e-5:
            stable = False
        else:
            stable = True
        stable_poles[:,i//skip_num] = stable
 
for i in range(stable_poles.shape[1]-7):
    if not np.any(stable_poles[0,i:i+7]):
        stable_i = i
        break
    
####### Results ##########

## Logging Final Guess to Real Value Percentage Error
print(100*np.abs(1-np.mean(guesses[2,:])/true_value))
print(100*np.abs(1-np.mean(guesses[2,:stable_i])/true_value), stable_i)

## Plotting Parameter Prediction over Time
plt.plot(t_shaved[0,:stable_i].T/60/60/24, guesses[2,:stable_i], label='Response')
plt.plot(t_shaved[0,:stable_i].T/60/60/24, parameter_guess[0,:stable_i], label='Rolling Mean Guess')
# plt.plot(t_shaved.T/60/60/24, true_value*np.ones((guesses.shape[1],1)), label='True Parameter Value')
plt.title('Parameter Observer Convergence, Noise')
plt.legend(loc='upper left')
plt.xlabel('Time (days)')
plt.ylabel('K Value')
plt.grid(True)
plt.show()

## Plotting Parameter Prediction over Time
plt.plot(t_shaved.T/60/60/24, guesses[2,:], label='Response')
plt.plot(t_shaved.T/60/60/24, parameter_guess[0,:], label='Rolling Mean Guess')
plt.plot(t_shaved.T/60/60/24, true_value*np.ones((guesses.shape[1],1)), label='True Parameter Value')
plt.title('Parameter Observer Convergence, Noise')
plt.legend(loc='upper left')
plt.xlabel('Time (days)')
plt.ylabel('K Value')
plt.grid(True)
plt.show()
