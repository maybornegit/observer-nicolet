#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 11:58:31 2024

@author: morganmayborne
"""

from sympy import *
import numpy as np

### Function inputing a input, parameter, and state list and recursing over them to find a A, B, and C matrix of interest
def getABC(u,x):
    def A_matrix(state_,x_dot):
        N = state_.shape[0]
        A = []
        for i in range(N):
            A.append([])
            for j in range(N):
                A[i].append(diff(x_dot[i],state_[j]))
                
        A = np.array(A)
        return A
    
    def B_matrix(input_,state_,x_dot):
        Ns = state_.shape[0]
        Ni = input_.shape[0]
        B = []
        for i in range(Ns):
            B.append([])
            for j in range(Ni):
                B[i].append(diff(x_dot[i],input_[j]))
                
        B = np.array(B)
        return B
    
    def C_matrix(output_,state_):
        Ns = state_.shape[0]
        No = output_.shape[0]
        C = []
        for i in range(No):
            C.append([])
            for j in range(Ns):
                C[i].append(diff(output_[i],state_[j]))
                
        C = np.array(C)
        return C
    
    def x_dot_def(x):
        #### Define Symbols
        M_cs, M_cv = symbols('M_cs M_cv')  # Core State Variables
        # M_dm, M_fm = symbols('M_dm M_fm')  # Core Output Variables
        I, T, C = symbols('I T C')         # Typical Input Variables
        
        k, v, a = symbols('k v a')         # Parameters - Sensitive
        eps, sig = symbols('eps sig')      # Parameters - Photosynthesis 
        c, th = symbols('c th')            # Parameters - Respiration
        gam, lam = symbols('gam lam')      # Parameters - Crop Composition
        s_p, s_g, b_p, b_g = symbols('s_p s_g b_p b_g') # Parameters - Attentuation
        C_st, T_st, Pi_v = symbols('C_st T_st Pi_v')    # Parameters - Constants
        beta, e_OMC, e_MMN = symbols('beta e_OMC e_MMN')# Parameters - Other
        
        #### Define Symbolic Functions
        C_cv = M_cv/(lam*M_cs)
        h_p = 1/(1+((b_g*Pi_v)/(gam*C_cv))**s_p)
        h_g = 1/(1+(((1-b_p)*Pi_v)/(gam*C_cv))**s_g)
        F_cav = ((eps*I*sig*(C-C_st))/(eps*I+sig*(C-C_st)))*(1-exp(-a*M_cs))*h_p
        F_cvs = v*k*exp(c*(T-T_st))*(1-exp(-a*M_cs))*h_g
        F_cm = v*exp(c*(T-T_st))
        F_cg = th*F_cvs
        M_dm = e_OMC*(M_cv+M_cs)+e_MMN*(lam*Pi_v*M_cs-gam*M_cv)/beta
        y = np.array([M_dm,(1000*lam*M_cs+M_dm)])
        
        x_dot = []
        M_cs, M_cv = symbols('M_cs M_cv')  # Core State Variables
        N = x.shape[0]
        for i in range(N):
            if x[i] == M_cv:
                x_dot.append(F_cav-h_g*F_cm-F_cg-F_cvs)
            elif x[i] == M_cs:
                x_dot.append(F_cvs-(1-h_g)*F_cm)
            else:
                x_dot.append(0)
        
        x_dot = np.array(x_dot)
        return x_dot
        
    gam, lam = symbols('gam lam')      # Parameters - Crop Composition
    C_st, T_st, Pi_v = symbols('C_st T_st Pi_v')    # Parameters - Constants
    beta, e_OMC, e_MMN = symbols('beta e_OMC e_MMN')# Parameters - Other
    
    M_dm = e_OMC*(M_cv+M_cs)+e_MMN*(lam*Pi_v*M_cs-gam*M_cv)/beta
    y = np.array([M_dm,(1000*lam*M_cs+M_dm)])     
    x_dot = x_dot_def(x)
    A = A_matrix(x,x_dot)
    B = B_matrix(u,x,x_dot)
    C = C_matrix(y,x)
    return A, B, C

def base_subs(A,B,C_,p,psym):
    Np = len(p)
    shapeA = A.shape
    shapeB = B.shape
    shapeC = C_.shape
    for i in range(Np):
        for j in range(shapeA[0]):
            for k in range(shapeA[1]):
                if not isinstance(A[j,k],float):
                    newA = A[j,k].subs(psym[i],p[i])
                    if isinstance(newA,Float):
                        A[j,k] = float(newA)
                    else:
                        A[j,k] = newA
                
        for j in range(shapeB[0]):
            for k in range(shapeB[1]):
                if not isinstance(B[j,k],float):
                    newB = B[j,k].subs(psym[i],p[i])
                    if isinstance(newB,Float):
                        B[j,k] = float(newB)
                    else:
                        B[j,k] = newB
                
        for j in range(shapeC[0]):
            for k in range(shapeC[1]):
                if not isinstance(C_[j,k],float):
                    newC = C_[j,k].subs(psym[i],p[i])
                    if isinstance(newC,Float):
                        C_[j,k] = float(newC)
                    else:
                        C_[j,k] = newC
    return A,B,C_

def h(xsym,psym,x,p):
    M_cs, M_cv = symbols('M_cs M_cv')  # Core State Variables
    # M_dm, M_fm = symbols('M_dm M_fm')  # Core Output Variables
    I, T, C = symbols('I T C')         # Typical Input Variables
    
    k, v, a = symbols('k v a')         # Parameters - Sensitive
    eps, sig = symbols('eps sig')      # Parameters - Photosynthesis 
    c, th = symbols('c th')            # Parameters - Respiration
    gam, lam = symbols('gam lam')      # Parameters - Crop Composition
    s_p, s_g, b_p, b_g = symbols('s_p s_g b_p b_g') # Parameters - Attentuation
    C_st, T_st, Pi_v = symbols('C_st T_st Pi_v')    # Parameters - Constants
    beta, e_OMC, e_MMN = symbols('beta e_OMC e_MMN')# Parameters - Other
    
    M_dm = e_OMC*(M_cv+M_cs)+e_MMN*(lam*Pi_v*M_cs-gam*M_cv)/beta
    y = np.array([M_dm,(1000*lam*M_cs+M_dm)])
    Np, Nx = len(p), len(x)
    
    for i in range(Np):
        y[0] = y[0].subs(psym[i],p[i])
        y[1] = y[1].subs(psym[i],p[i])
        
    for i in range(Nx):
        y[0] = y[0].subs(xsym[i],x[i])
        y[1] = y[1].subs(xsym[i],x[i])
    
    return y.astype(np.float64)
    
def f(xsym,usym,psym,x,u,p,i):
    M_cs, M_cv = symbols('M_cs M_cv')  # Core State Variables
    # M_dm, M_fm = symbols('M_dm M_fm')  # Core Output Variables
    I, T, C = symbols('I T C')         # Typical Input Variables
    
    k, v, a = symbols('k v a')         # Parameters - Sensitive
    eps, sig = symbols('eps sig')      # Parameters - Photosynthesis 
    c, th = symbols('c th')            # Parameters - Respiration
    gam, lam = symbols('gam lam')      # Parameters - Crop Composition
    s_p, s_g, b_p, b_g = symbols('s_p s_g b_p b_g') # Parameters - Attentuation
    C_st, T_st, Pi_v = symbols('C_st T_st Pi_v')    # Parameters - Constants
    beta, e_OMC, e_MMN = symbols('beta e_OMC e_MMN')# Parameters - Other
    
    C_cv = M_cv/(lam*M_cs)
    h_p = 1/(1+((1-b_p)/(gam*C_cv))**(s_p))
    h_g = 1/(1+((b_g*Pi_v)/(gam*C_cv))**(s_g))
    F_cav = ((eps*I*sig*(C-C_st))/((eps*I)+(sig*(C-C_st))))*(1-exp(-a*M_cs))*h_p
    F_cvs = v*k*exp(c*(T-T_st))*(1-exp(-a*M_cs))*h_g
    F_cm = k*exp(c*(T-T_st))*M_cs
    F_cg = th*F_cvs
    
    x_dot = np.array([F_cav-h_g*F_cm-F_cg-F_cvs,F_cvs-(1-h_g)*F_cm])
    Np, Nx, Nu = len(p), len(x), len(u)
    for i in range(Np):
        x_dot[0] = x_dot[0].subs(psym[i],p[i])
        x_dot[1] = x_dot[1].subs(psym[i],p[i])
        
    for i in range(Nx):
        x_dot[0] = x_dot[0].subs(xsym[i],x[i])
        x_dot[1] = x_dot[1].subs(xsym[i],x[i])
    
    for i in range(Nu):
        x_dot[0] = x_dot[0].subs(usym[i],u[i])
        x_dot[1] = x_dot[1].subs(usym[i],u[i])
    
    return x_dot.astype(np.float64) 
    
if __name__ == '__main__':
    #### Define Symbols
    M_cs, M_cv = symbols('M_cs M_cv')  # Core State Variables
    I, T, C = symbols('I T C')         # Typical Input Variables
    M_cs, M_cv = symbols('M_cs M_cv')  # Core State Variables
    # M_dm, M_fm = symbols('M_dm M_fm')  # Core Output Variables
    I, T, C = symbols('I T C')         # Typical Input Variables
    
    k, v, a = symbols('k v a')         # Parameters - Sensitive
    eps, sig = symbols('eps sig')      # Parameters - Photosynthesis 
    c, th = symbols('c th')            # Parameters - Respiration
    gam, lam = symbols('gam lam')      # Parameters - Crop Composition
    s_p, s_g, b_p, b_g = symbols('s_p s_g b_p b_g') # Parameters - Attentuation
    C_st, T_st, Pi_v = symbols('C_st T_st Pi_v')    # Parameters - Constants
    beta, e_OMC, e_MMN = symbols('beta e_OMC e_MMN')# Parameters - Other

    ## Parameter Array (see above for indices) ##
    p = [0.055, 1.4e-3, 0.0011, .5, 0.8, 580.0, 1/1200, 0.61, 10.0, 0.0693, 20.0, 0.3, 22.1*.5, .2, 10.0, 0.03, 0.148, 6.0,.35e-6]
    psym = np.array([eps, sig, C_st, a, b_p, Pi_v, lam, gam, s_p, c, T_st, th, v, b_g, s_g, e_OMC, e_MMN, beta,k])
    
    #### Get ABC Matrices
    u = np.array([T,C]).T
    x = np.array([M_cv,M_cs,I]).T
    A,B,C = getABC(u,x)
    A,B,C = base_subs(A,B,C,p,psym)
    # print(A,B,C)
    
    x0 = np.array([0.007, 0.0671,175.0*2.1e-5])  # Initial condition (M_cv, M_cs)
    u_ = np.array([28.0,450])  # Input (I [W/m2],T[C],C_co2[ppm])
    u_ *= np.array([1,.0195/450]) # Conversion to Model Units
    print(h(x,psym,x0,p))
    print(f(x,u,psym,x0,u_,p,0))
    ## Something is incorrect about the calculation of f