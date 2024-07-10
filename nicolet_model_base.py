#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 10:34:10 2024

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


####### Functions for the Nicolet Model ##########

## Core Function Solvers ##

def runge_kutta_4(f, x0, u, p, dt, iter_):
    '''
    Runge-Kutta 4 First-Order DiffEq Estimation for State Variables
    
    Parameters:
    f : function that computes the derivative at the current state
    x0 : Current state [M_cv, M_cs]
    u : Current input parameters [I, T, C_co2]
    p : NICOLET Model Parameter List [see list below for specific elements]
    dt : Expected Jump in Time to Approximate (in sec)

    Returns: 
        
    Estimated State at t+dt

    '''
    if iter_ < len(u)-1 and u.shape[0] != 3:
        iter_check = True
        u_mid = np.mean(u[iter_:(iter_+2),:], axis=0).reshape((1,-1))
    elif u.shape[0] != 3:
        iter_check = False
        u_mid = u[-1].reshape((1,-1))
        
    k1 = f(x0, u, p, iter_)
    k2 = f(x0 + 0.5*dt*k1, u, p, iter_)
    k3 = f(x0 + 0.5*dt*k2, u, p, iter_)
    k4 = f(x0 + dt*k3, u, p, iter_)
    next_state = x0 + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
    return next_state

def solve_system(f, h, x0, u, p, t_span, dt):
    '''
    Computes Core trajectories of NICOLET Model; first-order diffeq solver
    
    Parameters:
    f : Function Defining State Derivatives (req. x, u, p)
    h : Function for the Outputs (Biomass) (req. x, p)
    x0 : Initial State
    u : Input Parameters (currently constant throughout simulation)
    t_span : Defines Beginning / End Times for Trajectory Simulation
    dt: Time Delta for State Prediction (likely unneces)

    Returns: 
    t: Time Array
    x_traj: State Variable Array
    y_traj: Output Variable Array

    '''
    t = np.arange(t_span[0], t_span[1], dt)
    num_steps = len(t)
    x_traj = np.zeros((len(x0), num_steps))
    y_traj = np.zeros((len(h(x0, p)), num_steps))
    x_traj[:, 0] = x0
    

    for i in range(1, num_steps):
        if u.shape[0] == 3:
            x_traj[:, i] = runge_kutta_4(f, x_traj[:, i-1], u, p, dt, 0)
            y_traj[:, i] = h(x_traj[:, i], p)
        else:
            x_traj[:, i] = runge_kutta_4(f, x_traj[:, i-1], u, p, dt, i)
            y_traj[:, i] = h(x_traj[:, i], p)
    
    return t, x_traj, y_traj

def run_model(f, h, x0, u, p, t_span, dt, title, file_id='Regular_Test'):
    '''
    Multi-function wrapper for visualizing and saving model output, with capabilities for:
        (1) Graphing the trajectories for output variables
        (2) Saving t, x, y, and u data into .npy files
    
    Parameters:
    f : function that computes the current state derivative
    h : function that computes the current output variables
    x0 : Current state [M_cv, M_cs]
    u : Current input parameters [I, T, C_co2]
    p : NICOLET Model Parameter List [see list below for specific elements]
    t_span : 
    dt : Expected Jump in Time to Approximate (in sec)
    title: Plot Title
    file_id : If calibrated, output file identification name

    Returns: 
        
    None
    
    '''
    ## Use solve_system to compute trajectories for state and output
    t, x_traj, y_traj = solve_system(f, h, x0, u, p, t_span, dt)
         
    ## Output trajectories into .npy files (change folder variable)
    # folder = '/Users/morganmayborne/MSR General/Linear_observer_npy/'
    # timestamp =  datetime.now().strftime("%Y_%m_%d_%h_%M_%S")
    # file_x= folder+timestamp+'_x_data_'+file_id+'.npy'
    # file_y= folder+timestamp+'_y_data_'+file_id+'.npy'
    # file_t= folder+timestamp+'_t_data_'+file_id+'.npy'
    # file_u= folder+timestamp+'_u_data_'+file_id+'.npy'
    # np.save(file_x, x_traj)
    # np.save(file_y, y_traj)
    # np.save(file_t, t)
    # np.save(file_u, u)
    
    ## Plotting either state or output variables
    #plt.plot(t, x_traj[:,:].T, label=['M_cv', 'M_cs'])
    plt.plot(t/60/60/24, y_traj[:2,:].T,label=['M_dm', 'M_fm'])
    plt.title(title)
    plt.xlabel('Time (days)')
    plt.ylabel('Weight (kg m-2)')
    plt.legend()
    plt.grid(True)
    plt.show()
    return None

## State Space Function (f - derivative of state, h - output) ##

def f(x, u, p, iter_):  
    '''
    Function Defining the State Derivatives; for a specific time t
    
    Parameters:
    x: State Variables [M_cv, M_cs]
    u: Input Parameters [I, T, C_co2]
    p: NICOLET Model Parameters
            
    Returns: 
    xdot: Derivative Array for State Variables [M_cv, M_cs]

    '''
    def h_g(x,p):
        '''
        h_g Function; Source Depletion Switching Inhibition Function
        
        Parameters:
        x: State Variables [M_cv, M_cs]
        p: NICOLET Model Parameters
                
        Returns: Function Result

        '''
        pi_v = p[5]
        lambda_ = p[6]
        gamma = p[7]
        b_g = p[14]
        s_g = p[15]
        
        M_cv = x[0]
        M_cs = x[1]
        
        C_cv = M_cv/lambda_/M_cs
        return 1/(1+((b_g*pi_v)/(gamma*C_cv))**s_g)
    
    if u.shape[0] != 3:
        u = u[iter_, :]
    xdot = np.zeros((1,2)).squeeze()
    #print(F_cav(x, u, p).shape, F_cm(x, u, p).shape, F_cg(x, u, p).shape, F_cvs(x, u, p).shape)
    #print(F_cav(x, u, p), h_g(x,p),F_cm(x, u, p),F_cg(x, u, p), F_cvs(x, u, p))
    xdot[0] = F_cav(x, u, p) - h_g(x,p)*F_cm(x, u, p) - F_cg(x, u, p) - F_cvs(x, u, p)
    xdot[1] = F_cvs(x,u,p) - (1-h_g(x,p))*F_cm(x, u, p)
    return xdot

def h(x, p):
    '''
    Function Defining the Outputs; for a specific time t
    
    Parameters:
    x: State Variables [M_cv, M_cs]
    p: NICOLET Model Parameters
            
    Returns: 
    y: Output Array [M_dm, M_fm]

    '''
    pi_v = p[5]
    lambda_ = p[6]
    gamma = p[7]
    beta = p[18]
    
    eta_NO3N = p[19]
    
    M_cv = x[0]
    M_cs = x[1]
    
    C_cv = M_cv/lambda_/M_cs
    C_nv = (pi_v-(gamma*C_cv))/beta

    y = np.zeros((1,3)).squeeze()
    y[0] = M_dm(x,p)
    y[1] = (1000*lambda_*M_cs)+y[0]
    C_NO3N = C_nv*(1-(M_dm(x,p)/y[1]))/1000
    y[2] = 1e6*eta_NO3N*C_NO3N
    return y

## State / Carbon Flow Functions ##

def F_cav(x, u, p):
    '''
    Photosynthetic Assimilation; for a specific time t
    
    Parameters:
    x: State Variables [M_cv, M_cs]
    u: Input Parameters [I, T, C_co2]
    p: NICOLET Model Parameters
            
    Returns: 
    F_cav: Photosynthetic Assimilation Metric; in mol[C]/m2/s

    '''
    eps = p[0]
    sigma = p[1]
    co2_baseline = p[2]
    a = p[3]
    b_p = p[4]
    pi_v = p[5]
    lambda_ = p[6]
    gamma = p[7]
    s_p = p[8]
    
    I = u[0]
    C_co2 = u[2]
    
    M_cv = x[0]
    M_cs = x[1]
    
    C_cv = M_cv/lambda_/M_cs
    
    p_ICca = (eps*I*sigma*(C_co2-co2_baseline))/((eps*I)+(sigma*(C_co2-co2_baseline)))
    f_Mcs = 1-np.exp(-a*M_cs)
    h_pCcv = 1/(1+(((1-b_p)*pi_v)/(pi_v*(gamma*C_cv)))**(s_p))
    
    F_cav = p_ICca*f_Mcs*h_pCcv
    return F_cav

def F_cm(x, u, p):
    '''
    Maintenance Respiration; for a specific time t
    
    Parameters:
    x: State Variables [M_cv, M_cs]
    u: Input Parameters [I, T, C_co2]
    p: NICOLET Model Parameters
            
    Returns: 
    F_cm_: Maintenance Respiration Metric; in mol[C]/m2/s

    '''
    K = p[9]
    c = p[10]
    t_baseline = p[11]
    T = u[1]
    M_cs = x[1]
    
    e_T = K*np.exp(c*(T-t_baseline))
    F_cm_ = e_T*M_cs
    
    return F_cm_

def F_cg(x, u, p):
    '''
    Growth Respiration; for a specific time t
    
    Parameters:
    x: State Variables [M_cv, M_cs]
    u: Input Parameters [I, T, C_co2]
    p: NICOLET Model Parameters
            
    Returns: 
    F_cg: Growth Respiration Metric; in mol[C]/m2/s

    '''
    theta = p[12]
    F_cg = theta*F_cvs(x,u,p)
    return F_cg

def F_cvs(x, u, p):
    '''
    Growth; for a specific time t
    
    Parameters:
    x: State Variables [M_cv, M_cs]
    u: Input Parameters [I, T, C_co2]
    p: NICOLET Model Parameters
            
    Returns: 
    F_cvs: Growth Metric; in mol[C]/m2/s

    '''
    a = p[3]
    pi_v = p[5]
    lambda_ = p[6]
    gamma = p[7]
    K = p[9]
    c = p[10]
    t_baseline = p[11]
    v = p[13]
    b_g = p[14]
    s_g = p[15]
    
    T = u[1]
    
    M_cv = x[0]
    M_cs = x[1]
    
    C_cv = M_cv/lambda_/M_cs
    
    g_T = v*K*np.exp(c*(T-t_baseline)) #######
    f_Mcs = (1-np.exp(-a*M_cs))
    hg_Ccv = 1/(1+((b_g*pi_v)/(gamma*C_cv))**s_g)
    
    F_cvs = g_T*f_Mcs*hg_Ccv
    return F_cvs

## Output Function(s) ##

def M_dm(x, p):
    '''
    Dry Biomass Output Function; for a specific time t
    
    Parameters:
    x: State Variables [M_cv, M_cs]
    p: NICOLET Model Parameters
            
    Returns: 
    M_dm: Dry Biomass; in kg/m2

    '''
    pi_v = p[5]
    lambda_ = p[6]
    gamma = p[7]
    eta_OMC = p[16]
    eta_MMN = p[17]
    beta = p[18]
    
    M_cv = x[0]
    M_cs = x[1]
    
    term_1 = eta_OMC*(M_cv+M_cs)
    term_2_1 = eta_MMN*(lambda_*pi_v*M_cs)/beta
    term_2_2 = -eta_MMN*gamma*M_cv/beta
    M_dm = term_1+term_2_1+term_2_2
    return M_dm

## Other Helper Function ##

def p_(u,p):
    '''
    Uninhibited Photosynthesis Rate Function; for a specific time t
    
    Parameters:
    u: Input Parameters [I, T, C_co2]
    p: NICOLET Model Parameters
            
    Returns: 
    p_ICca: Uninhibited Photosynthesis Rate; in mol[C]/m2/s

    '''
    eps = p[0]
    sigma = p[1]
    co2_baseline = p[2]
    
    I = u[0]
    C_co2 = u[2]
    
    p_ICca = (eps*I*sigma*(C_co2-co2_baseline))/((eps*I)+(sigma*(C_co2-co2_baseline)))
    return p_ICca

def f_(x,p):
    '''
    Canopy Closure Reduction Function; for a specific time t
    
    Parameters:
    u: Input Parameters [I, T, C_co2]
    p: NICOLET Model Parameters
            
    Returns: 
    f_Mcs: Canopy Closure Reduction Result; dimensionless

    '''
    a = p[3]
    M_cs = x[1]
    
    f_Mcs = 1-np.exp(-a*M_cs)
    return f_Mcs

def e_(x,u,p):
    '''
    Specific Maintenance Respiration Function; for a specific time t
    
    Parameters:
    x: State Variables [M_cv, M_cs]
    u: Input Parameters [I, T, C_co2]
    p: NICOLET Model Parameters
            
    Returns: 
    e_: Specific Maintenance Respiration Metric; in 1/s

    '''
    c = p[10]
    t_baseline = p[11]
    k = x[2]
    T = u[1]
    
    e_ = k*np.exp(c*(T-t_baseline))
    return e_

def g_(x,u,p):
    '''
    Maximum Growth Rate Function; for a specific time t
    
    Parameters:
    x: State Variables [M_cv, M_cs]
    u: Input Parameters [I, T, C_co2]
    p: NICOLET Model Parameters
            
    Returns: 
    g_: Maximum Growth Rate Metric; in mol[C]/m2/s

    '''
    c = p[10]
    t_baseline = p[11]
    v = p[13]
    k = x[2]
    T = u[1]
    
    g_ = v*k*np.exp(c*(T-t_baseline))
    return g_

def h_p(x,p):
    '''
    Photosynthesis Inhibition Functionn; for a specific time t
    
    Parameters:
    x: State Variables [M_cv, M_cs]
    p: NICOLET Model Parameters
            
    Returns: 
    h_pCcv: Photosynthesis Inhibition Function; dimensionless

    '''
    b_p = p[4]
    pi_v = p[5]
    lambda_ = p[6]
    gamma = p[7]
    s_p = p[8]
    
    M_cv = x[0]
    M_cs = x[1]
    C_cv = M_cv/lambda_/M_cs
    
    h_pCcv = 1/(1+(((1-b_p)*pi_v)/(pi_v*(gamma*C_cv)))**(s_p))
    return h_pCcv

def h_g(x,p):
    '''
    Source Depletion Switching Functionn; for a specific time t
    
    Parameters:
    x: State Variables [M_cv, M_cs]
    p: NICOLET Model Parameters
            
    Returns: 
    h_gCcv: Source Depletion Switching Function; dimensionless

    '''
    pi_v = p[5]
    lambda_ = p[6]
    gamma = p[7]
    b_g = p[14]
    s_g = p[15]
    
    M_cv = x[0]
    M_cs = x[1]
    
    C_cv = M_cv/lambda_/M_cs
    h_gCcv = 1/(1+((b_g*pi_v)/(gamma*C_cv))**s_g)
    return h_gCcv

## Linearization Helper Functions (See A matrix below) ##

def dh_p_mcv(x,p):
    '''
    Partial Derivative of Photosynthesis Inhibition Function w.r.t non-structural carbon; for a specific time t
    
    Parameters:
    x: State Variables [M_cv, M_cs]
    p: NICOLET Model Parameters
            
    Returns: 
    dh_p: P.D of h_p function w.r.t M_cv; dimensionless

    '''
    b_p = p[4] 
    lambda_ = p[6] 
    gamma = p[7] 
    s_p = p[8] 
    M_cv = x[0]
    M_cs = x[1]
    
    dh_p = s_p*((M_cs*lambda_*(1-b_p))/(M_cv*gamma))**s_p*(h_p(x, p)**2)/M_cv
    return dh_p

def dh_g_mcv(x,p):
    '''
    Partial Derivative of Source Depletion Switching Function w.r.t non-structural carbon; for a specific time t
    
    Parameters:
    x: State Variables [M_cv, M_cs]
    p: NICOLET Model Parameters
            
    Returns: 
    dh_g: P.D of h_g function w.r.t M_cv; dimensionless

    '''
    pi_v = p[5]
    lambda_ = p[6]
    gamma = p[7]
    b_g = p[14]
    s_g = p[15]
    
    M_cv = x[0]
    M_cs = x[1]
    
    dh_g = s_g*((M_cs*lambda_*pi_v*b_g)/(M_cv*gamma))**s_g*(h_g(x, p)**2)/M_cv
    return dh_g

def dh_p_mcs(x,p):
    '''
    Partial Derivative of Photosynthesis Inhibition Function w.r.t structural carbon; for a specific time t
    
    Parameters:
    x: State Variables [M_cv, M_cs]
    p: NICOLET Model Parameters
            
    Returns: 
    dh_p: P.D of h_p function w.r.t M_cs; dimensionless

    '''
    b_p = p[4] 
    lambda_ = p[6] 
    gamma = p[7] 
    s_p = p[8] 
    
    M_cv = x[0]
    M_cs = x[1]
    
    dh_p = -s_p*((M_cs*lambda_*(1-b_p))/(M_cv*gamma))**s_p*(h_p(x, p)**2)/M_cs
    return dh_p

def dh_g_mcs(x,p):
    '''
    Partial Derivative of Source Depletion Switching Function w.r.t structural carbon; for a specific time t
    
    Parameters:
    x: State Variables [M_cv, M_cs]
    p: NICOLET Model Parameters
            
    Returns: 
    dh_g: P.D of h_g function w.r.t M_cs; dimensionless

    '''
    pi_v = p[5]
    lambda_ = p[6]
    gamma = p[7]
    b_g = p[14]
    s_g = p[15]
    
    M_cv = x[0]
    M_cs = x[1]
    
    dh_g = -s_g*((M_cs*lambda_*pi_v*b_g)/(M_cv*gamma))**s_g*(h_g(x, p)**2)/M_cs
    return dh_g

# Solve the system
def run_model(f, h, x0, u, p, t_span, dt, title, file_id='Regular_Test'):
    t, x_traj, y_traj = solve_system(f, h, x0, u, p, t_span, dt)
    
    # # Export smaller data
    # t_ = []
    # y = []
    # x = []
    
    # for i in range(len(t)):
    #     if i % 500 == 0:
    #         t_.append(t[i])
    #         y.append(y_traj[:2,i])
    #         x.append(x_traj[:,i])
            
    # print(np.array(t_))
    # print(np.array(x))
    # print(np.array(y))
    set_point = y_traj[1,-1]
    done = False
    for i in range(len(y_traj[1,:])):
        if y_traj[1,i] >= .93*set_point and y_traj[1,i] <= 1.07*set_point and (not done):
            done = True
         
    # folder = '/Users/morganmayborne/MSR General/Linear_observer_npy/'
    # timestamp =  datetime.now().strftime("%Y_%m_%d_%h_%M_%S")
    # file_x= folder+timestamp+'_x_data_'+file_id+'.npy'
    # file_y= folder+timestamp+'_y_data_'+file_id+'.npy'
    # file_t= folder+timestamp+'_t_data_'+file_id+'.npy'
    # file_u= folder+timestamp+'_u_data_'+file_id+'.npy'
    # np.save(file_x, x_traj)
    # np.save(file_y, y_traj)
    # np.save(file_t, t)
    # np.save(file_u, u)
    
    #plt.plot(t, x_traj[:,:].T, label=['M_cv', 'M_cs'])
    plt.plot(t/60/60/24, y_traj[:2,:].T,label=['M_dm', 'M_fm'])
    plt.title(title)
    plt.xlabel('Time (days)')
    plt.ylabel('Weight (kg m-2)')
    plt.legend()
    plt.grid(True)
    plt.show()
    return None

if __name__ == "__main__":
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
    
    ## Initialize Test Input Array ##
    u = u*np.ones((t_span[1]//dt,3)) # Expanding Input to Fill Full Trajectory
    u[:,0] *= np.array([(1 if i*3600 % 86400 < 36000 else 0) for i in range(t_span[1]//dt)]) # Input PAR - Step Function
    u[:,1] += 2.0*np.cos(2*np.pi*(t_array-3600)/86400)  # Input Temp. - Sinusodial Daily Changes
    u[:,2] += np.random.normal(0,1.5,t_span[1]//dt)*.22*u[:,2]  # Input Co2 - Gaussian Changes
    
    ### Run the NICOLET Model (Un-comment run_model for graphical representations)
    run_model(f, h, x0, u, p, t_span, dt, 'Base Case', file_id='k_IC_test')
    t_traj, x_traj, y_traj = solve_system(f, h, x0, u, p, t_span, dt) # Basic run for model, no graphs


