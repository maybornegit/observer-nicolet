#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 10:38:36 2024

@author: morganmayborne
"""

import numpy as np
import matplotlib.pyplot as plt

y_traj = np.load('./dry_mass.npy')
y_traj[1,:] /= 20
t_traj = np.load('./dry_mass_time.npy')

plt.figure(0)
plt.plot(t_traj/60/60/24,y_traj[0,:].T, label="Simulation M_dm")
plt.plot(t_traj/60/60/24,y_traj[1,:].T, label="M_fm * DFW")
plt.title('Fixed DFW Checking')
plt.xlabel('Time (Days)')
plt.ylabel('Fresh Biomass (in g)')
plt.legend()
plt.grid(True)
plt.show()
