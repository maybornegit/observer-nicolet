#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 13:51:08 2024

@author: morganmayborne
"""
import random, csv
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from nicolet_model_base import basic_test_w_lighting, residual_function_kva, residual_function_one_param, residual_function_two_params
from generate_trajs import traj_analysis
from scipy.optimize import minimize
import pandas as pd
import seaborn as sns

plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pacoty.mplstyle')

class Minimizer():
    def __init__(self,init_guess, bounds,train_trajs, test_trajs,lighting, spec='K', huber=False, repress_print=False):
        self.init_guess = np.array(init_guess)
        self.bounds = np.array(bounds)
        self.type = spec
        self.huber = huber
        self.huber_delta = 0.05
        self.param_names = ['Maintanence Respiration (K)', 'Growth Rate Coefficient (V)', 'Leaf Area Closure (A)']
        self.train_trajs = train_trajs
        self.test_trajs = test_trajs
        self.light_conds = lighting
        self.trajs = self.train_trajs+self.test_trajs
        self.repress_print = repress_print

        self.guess = []
        self.guess_param_setup()

        self.dt = 1

        self.err_by_day = []  #### identify the day and error amount for error bar plotting
        self.true_ = []       #### find ground truth values for average
        self.sim = []         #### find simulation values for average

        self.history = []

        # self.inliers = [1,4,7,9,11,12,14,16,17,20,21,23,24,26,27,28,29,30,31,35,38,40,41,43,44,45,46,47]
        # self.midliers = [2,3,5,6,10,13,18,19,25,33,34,36,37,42,48]
        # self.outliers = [0,8,15,22,32,39,49]

        self.species = 1
        self.plant = 0
        self.param_order = [] # empty, except for K_V + K_V_A

        trajs = self.train_trajs[:]

        if spec == 'K':
            self.incl = np.array([1,0,0])
            self.species_args = [trajs,self.light_conds,self.dt, self.huber_delta, self.huber, spec]
        elif spec == 'V':
            self.incl = np.array([0,1,0])
            self.species_args = [trajs,self.light_conds,self.dt, self.huber_delta, self.huber, spec]
        elif spec == 'A':
            self.incl = np.array([0,0,1])
            self.species_args = [trajs,self.light_conds,self.dt, self.huber_delta, self.huber, spec]
        elif spec == 'KV':
            self.incl = np.array([1,1,0])
            self.species_args = [trajs,self.light_conds,self.dt, self.huber_delta, self.huber, spec]
        elif spec == 'VA':
            self.incl = np.array([0,1,1])
            self.species_args = [trajs,self.light_conds,self.dt, self.huber_delta, self.huber, spec]
        elif spec == 'KA':
            self.incl = np.array([1,0,1])
            self.species_args = [trajs,self.light_conds,self.dt, self.huber_delta, self.huber, spec]
        elif spec == 'KVA':
            self.incl = np.array([1,1,1])
            self.species_args = [trajs,self.light_conds,self.dt, self.huber_delta, self.huber, spec]
        elif spec == 'K_V_A':
            self.incl = np.array([1,0,0])
            self.species_args = [trajs,self.light_conds,self.dt, self.huber_delta, self.huber, 'K']
            self.param_order = ['V','A'] # start with k, then do these
        elif spec == 'K_V':
            self.incl = np.array([1,0,0])
            self.species_args = [trajs,self.light_conds,self.dt, self.huber_delta, self.huber, 'K']
            self.param_order = ['V'] # start with k, then do these
        elif spec == 'V_A':
            self.incl = np.array([0,1,0])
            self.species_args = [trajs,self.light_conds,self.dt, self.huber_delta, self.huber, 'V']
            self.param_order = ['A'] # start with k, then do these
        elif spec == 'None':
            self.species = 0

        if spec != 'None':
            for guess in self.init_guess[self.incl==0]:
                self.species_args.append(guess)
        
    # def light_reading(self, filename):
    #     with open(filename,'r') as f:
    #         reader = csv.reader(f)
    #         read_list = list(reader)
    #         self.light_conds = read_list[0]
    #     self.light_conds = np.array(self.light_conds, dtype=np.float16)
    #     return None
    
    def guess_param_setup(self):
        for _ in range(len(self.trajs)):
            self.guess.append([self.init_guess[0],self.init_guess[1],self.init_guess[2],1e6])
    
    def spec_min(self, plt_history=False):
        def callback(x):
            self.history.append([x,self.res_func_s(x)])

        if not self.species:
            return None

        init_guess = np.array(self.init_guess[self.incl==1])
        bounds = self.bounds[self.incl==1,:]

        self.history.append([init_guess,self.res_func_s(init_guess)])
        result = minimize(self.res_func_s, init_guess, method='Nelder-Mead', bounds=bounds, callback=callback, tol=1e-9)
        
        for i in range(len(self.guess)):
            count_incl = 0
            for j in range(4):
                if j == 3:
                    break
                if self.incl[j]:
                    self.guess[i][j] = result.x[count_incl]
                    count_incl += 1

        value = np.array([guess[0] for guess in self.history])
        cost = np.array([guess[1] for guess in self.history])
        if plt_history:
            plt_count = 0
            for i, incl in enumerate(self.incl):
                if not incl:
                    continue
                plt.figure(30+i)
                plt.plot(np.arange(0,value.shape[0]), value[:,plt_count], label="Parameter")
                plt.title(self.param_names[i]+" Estimation")
                plt.xlabel('Iteration (#)')
                plt.ylabel('Plant Respiration Parameter (1/s)')
                plt.legend()
                plt.grid(True)
                plt_count += 1

            plt.figure(35)
            plt.plot(np.arange(0,value.shape[0]), cost, label="Residual Value")
            plt.title("Parameter - Cost Function")
            plt.xlabel('Iteration (#)')
            plt.ylabel('Residual - Squared Distance (kg m-2)')
            plt.legend()
            plt.grid(True)
        return result
    
    def plant_min(self, i):
        if not self.plant:
            return None

        lght = [self.light_conds[i]]
        traj = [self.trajs[i]]
        args = [traj,lght,self.dt, 0, False]
        for guess in np.array(self.guess)[0,:3][self.incl==1]:
            args.append(guess)
        init_guess = np.array(self.init_guess[self.incl==0])
        bounds = self.bounds[self.incl==0,:]

        result = minimize(self.res_func_p, init_guess,args=(args,), method='Nelder-Mead', bounds=bounds)

        count_incl = 0
        for j in range(4):
            if j == 3:
                self.guess[i][j] = result.fun
                break
            if self.incl[j]:
                self.guess[i][j] = result.x[count_incl]
                count_incl += 1
        return result
    
    def optimize_loop(self,prepare_plot=False, plt_history=False):
        if not self.repress_print:
            print('------------',self.type, 'Species Optimization Initialized ------------')
        result_k = self.spec_min(plt_history=plt_history)
        if not self.repress_print:
            print('Initial Guess:',self.init_guess)
        if self.param_order:
            for param in self.param_order:
                if result_k != None:
                    if not self.repress_print:
                        print('Minimized Parameters: ',result_k.x)  # Optimal values
                        print('Minimized Function: ',result_k.fun)  # Minimum function value

                self.change_spec(param)
                result_k = self.spec_min(plt_history=plt_history)
                if not self.repress_print:
                    print('Initial Guess:',self.init_guess)
        
        if result_k != None and not self.repress_print:
            print('Minimized Parameters: ',result_k.x)  # Optimal values
            print('Minimized Function: ',result_k.fun)  # Minimum function value

        if not self.repress_print:
            print('------------',self.type, 'Species Optimization Complete ------------')
        residual = 0
        self.err_by_day = []  #### identify the day and error amount for error bar plotting
        self.true_ = []       #### find ground truth values for average
        self.sim = []         #### find simulation values for average

        start = 0
        end = len(self.train_trajs)
        
        if not self.repress_print:
            print('------------',self.type, 'Plant Optimization Initialized ------------')
        for i in tqdm(range(start, end), disable=self.repress_print):
            lighting = self.light_conds[i]

            # Nelder-Mead optimization
            # result = minimize(residual_function_va, np.array([v,a]),args=([new_trajs[i],lighting,4,result_k.x[0]],), method='Nelder-Mead', bounds=bounds_va)
            result = self.plant_min(i)
            # print('Minimized Parameters: ',result.x)  # Optimal values
            # print(i, ':','Minimized Function: ',result.fun)  # Minimum function value
            if result != None:
                residual += result.fun

            t_traj, _, y_traj = basic_test_w_lighting(lighting, self.guess[i][0], self.guess[i][1], self.guess[i][2], self.dt)

            lim = []
            for k,t in enumerate(t_traj):
                ## Find limits for plotting
                if t/60/60/24 >= np.min(self.trajs[i][0]) and len(lim) == 0 and prepare_plot:
                    lim.append(k)
                elif t/60/60/24 >= np.max(self.trajs[i][0]) and len(lim) == 1 and prepare_plot:
                    lim.append(k)
                
                ## Find the Day Marker in the Trajectory
                if t/60/60/24 in self.trajs[i][0]:
                    idx = list(self.trajs[i][0]).index(t/60/60/24)
                    if self.trajs[i][0][idx] not in [err[0] for err in self.err_by_day]:
                        self.err_by_day.append([self.trajs[i][0][idx],(y_traj[1][k]-self.trajs[i][1][idx])**2])
                        self.true_.append([self.trajs[i][0][idx],self.trajs[i][1][idx]])
                        self.sim.append([self.trajs[i][0][idx],y_traj[1,k]])
                    else:
                        for e, err in enumerate(self.err_by_day):
                            if err[0] == self.trajs[i][0][idx]:
                                self.err_by_day[e].append((y_traj[1][k]-self.trajs[i][1][idx])**2)
                                self.true_.append([self.trajs[i][1][idx]])
                                self.sim.append([y_traj[1,k]])

            if prepare_plot:
                plt.figure(i)
                plt.plot(np.array(self.trajs[i][0],dtype=np.float64),np.array(self.trajs[i][1],dtype=np.float64),label='M_fm - Ground Truth')
                plt.plot(t_traj[lim[0]:lim[1]]/60/60/24, y_traj[1,lim[0]:lim[1]].T, label="M_fm - Simulation")
                plt.title("Calibrated Simulation")
                plt.xlabel('Time (days)')
                plt.ylabel('Weight (kg m-2)')
                plt.legend()
                plt.grid(True)
        if not self.repress_print: 
            print('------------',self.type, 'Plant Optimization Complete ------------')
        return residual
    
    def avg_and_error_plot(self, avg=True, error=True):
        self.err_by_day = sorted(self.err_by_day, key=lambda x: x[0])
        self.sim = sorted(self.sim, key=lambda x: x[0])
        self.true_ = sorted(self.true_, key=lambda x: x[0])

        errs = np.array([[err[0], np.mean(np.array((err[1:]))**2)] for err in self.err_by_day])
        sim = np.array([[s[0], np.mean(np.array((s[1:])))] for s in self.sim])
        true = np.array([[t_[0], np.mean(np.array((t_[1:])))] for t_ in self.true_])

        plt.figure(len(self.trajs)+10)
        if avg:
            plt.plot(true[:,0], true[:,1], label="Mean Weight - Ground Truth")
            plt.plot(sim[:,0], sim[:,1], label="Mean Weight - Simulation")
        if error:
            plt.plot(errs[:,0], errs[:,1], label="Errors")
        plt.title("Squared Error w/ Calibrated Simulation")
        plt.xlabel('Time (days)')
        plt.ylabel('Weight (kg m-2)')
        if avg:
            plt.legend()
        plt.grid(True)

    def testset_eval(self, prepare_plot=False):
        residual = 0
        residuals = []
        # inl_res = []
        # midl_res = []
        # outl_res = []
        self.err_by_day = []  #### identify the day and error amount for error bar plotting
        self.true_ = []       #### find ground truth values for average
        self.sim = []         #### find simulation values for average
        if not self.repress_print:
            print('------------ Parameter Evaluation Initialized ------------')

        for i in tqdm(range(len(self.train_trajs), len(self.trajs)), disable=self.repress_print):
            traj = self.trajs[i]
            lighting = self.light_conds[i]
            _, tmp_residuals = residual_function_kva(self.guess[i][:3], [[traj], [lighting], self.dt, self.huber_delta, self.huber])
            residuals = residuals + list(tmp_residuals)

            t_traj, _, y_traj = basic_test_w_lighting(lighting, self.guess[i][0], self.guess[i][1], self.guess[i][2], 1)
            lim = []
            traj_sim = []
            for k,t in enumerate(t_traj):
                ## Find limits for plotting
                if t/60/60/24 >= np.min(self.trajs[i][0]) and len(lim) == 0 and prepare_plot:
                    lim.append(k)
                elif t/60/60/24 >= np.max(self.trajs[i][0]) and len(lim) == 1 and prepare_plot:
                    lim.append(k)
                
                ## Find the Day Marker in the Trajectory
                if t/60/60/24 in self.trajs[i][0]:
                    idx = list(self.trajs[i][0]).index(t/60/60/24)
                    if self.trajs[i][0][idx] not in [err[0] for err in self.err_by_day]:
                        self.err_by_day.append([self.trajs[i][0][idx],(self.trajs[i][1][idx]-y_traj[1][k])])
                        self.true_.append([self.trajs[i][0][idx],self.trajs[i][1][idx]])
                        self.sim.append([self.trajs[i][0][idx],y_traj[1,k]])
                    else:
                        for e, err in enumerate(self.err_by_day):
                            if err[0] == self.trajs[i][0][idx]:
                                self.err_by_day[e].append((self.trajs[i][1][idx]-y_traj[1][k]))
                                self.true_.append([self.trajs[i][1][idx]])
                                self.sim.append([y_traj[1,k]])

            if prepare_plot:
                cutoff_plot = [[],[]]
                for j, t in enumerate(self.trajs[i][0]):
                    cutoff_plot[0].append(t)
                    cutoff_plot[1].append(self.trajs[i][1][j])
                plt.figure(i)
                
                if self.get_final_guess() == [4.0e-7,22.1,0.5]:
                    plt.plot(np.array(cutoff_plot[0],dtype=np.float64),np.array(cutoff_plot[1],dtype=np.float64),label='Ground Truth', marker='o')
                    plt.plot(t_traj[lim[0]:lim[1]]/60/60/24, y_traj[1,lim[0]:lim[1]].T, label="Baseline Simulation")
                else:
                    plt.plot(t_traj[lim[0]:lim[1]]/60/60/24, y_traj[1,lim[0]:lim[1]].T, label="Updated Simulation")
                plt.title("Calibrated Simulation")
                plt.xlabel('Time (days)')
                plt.ylabel('Weight (kg m-2)')
                plt.legend()
                plt.grid(True)

        residual = np.array(residuals).mean()
        std_residual = np.std(np.array(residuals))

        ## Violin Plot

        # Flatten the list of errors and create corresponding trajectory labels
        if prepare_plot and self.get_final_guess() != [4.0e-7,22.1,0.5]:
            self.err_by_day = sorted(self.err_by_day, key=lambda x: x[0])
            days = [(sublist[0],len(sublist)-1) for sublist in self.err_by_day]
            flat_errors = [sublist[1:] for sublist in self.err_by_day]
            flat_errors = [error for sublist in flat_errors for error in sublist]
            # print(days)

            trajectory_labels = [f'{int(days[i][0])}' for i in range(len(days)) for _ in range(days[i][1])]

            # Create a DataFrame
            df = pd.DataFrame({
                'error': flat_errors,
                'trajectory': trajectory_labels
            })

            # Create the violin plot
            plt.figure(figsize=(10, 6))
            sns.violinplot(x='trajectory', y='error', data=df, inner='point', density_norm='width')
            plt.axhline(y=0, color='red', linestyle='--', label='residual = 0')  # You can customize the color, linestyle, and label

            # Customize the plot
            plt.title('Error Distribution At Days of Measurement')
            plt.xlabel('Time (in days)')
            plt.ylabel('Residual Value (in kg m-2)')

            # Show the plot
            plt.tight_layout()
        # print(inl_res, midl_res, outl_res)
        if not self.repress_print:
            print("Residual per Measurement:", residual)
            print("Std. Resid. per Measurement:", std_residual)
            # print("Residual per Inlier:", np.array(inl_res).mean())
            # print("Residual per Midlier:",np.array(midl_res).mean())
            # print("Residual per Outlier:", np.array(outl_res).mean())
            print('------------ Parameter Evaluation Complete ------------\n')
        return residual, std_residual
 
    def res_func_s(self, x):
        ## Wrapper function for Residual of a Batch of Plants (species-level optimization)
        type_res = self.species_args[5]

        if type_res == 'K':
            return residual_function_one_param(x,self.species_args)[0]
        elif type_res == 'V':
            return residual_function_one_param(x,self.species_args)[0]
        elif type_res == 'A':
            return residual_function_one_param(x,self.species_args)[0]
        elif type_res == 'KV':
            return residual_function_two_params(x,self.species_args)[0]
        elif type_res == 'VA':
            return residual_function_two_params(x,self.species_args)[0]
        elif type_res == 'KA':
            return residual_function_two_params(x,self.species_args)[0]
        elif type_res == 'KVA':
            return residual_function_kva(x,self.species_args)[0]

    def res_func_p(self, x, args):
        ## Wrapper function for Residual of a Single Plant (plant-level optimization)
        if self.type == 'K':
            # return residual_function_va(x,args)
            raise NotImplementedError
        else:
            raise NotImplementedError

    def change_spec(self, new_type): 
        trajs = self.train_trajs

        if new_type == 'K':
            self.incl = np.array([1,0,0])
            self.species_args = [trajs,self.light_conds,self.dt, self.huber_delta, self.huber, new_type]
        elif new_type == 'V':
            self.incl = np.array([0,1,0])
            self.species_args = [trajs,self.light_conds,self.dt, self.huber_delta, self.huber, new_type]
            self.plant = 0
        elif new_type == 'A':
            self.incl = np.array([0,0,1])
            self.species_args = [trajs,self.light_conds,self.dt, self.huber_delta, self.huber, new_type]
            self.plant = 0
        elif new_type == 'KV':
            self.incl = np.array([1,1,0])
            self.species_args = [trajs,self.light_conds,self.dt, self.huber_delta, self.huber, new_type]
            self.plant = 0
        elif new_type == 'VA':
            self.incl = np.array([0,1,1])
            self.species_args = [trajs,self.light_conds,self.dt, self.huber_delta, self.huber, new_type]
            self.plant = 0
        elif new_type == 'KA':
            self.incl = np.array([1,0,1])
            self.species_args = [trajs,self.light_conds,self.dt, self.huber_delta, self.huber, new_type]
            self.plant = 0
        
        for guess in self.init_guess[self.incl==0]:
            self.species_args.append(guess)

        self.init_guess = np.array([self.guess[0][0],self.guess[0][1], self.guess[0][2]])
            
        return None

    def residual_all_traj(self, histogram=True):
        residuals = []
        for i in tqdm(range(len(self.trajs))):
            traj = self.trajs[i]
            lighting = self.light_conds[i]
            residual = residual_function_kva(self.guess[i][:3], [[traj], [lighting], 1, 0, False])

            residuals.append(residual)

        residuals = np.array(residuals)
        mean = residuals[:,1].mean()
        std = np.std(residuals[:,1])
        if histogram:
            plt.figure(100)
            r_histogram = plt.hist(residuals[:,1], bins=np.arange(0,1.2,0.025))
            plt.title("Histogram of Residuals") 
            plt.ylabel("Count")
            plt.xlabel("Average Residual / Measurement")
        return residuals

    def get_final_guess(self):
        return self.guess[0][:3]

def create_train_test(seed, filter, train_pct=.9):
    def pull_wgt_per_area(raw_trajs,factors,cutoffs):##### Find trajectories divided by area
        trajs = []
        in_conv = 0.0254
        for i in range(len(raw_trajs)):
            cur_traj = np.array(raw_trajs[i],dtype=np.float64)
            for j in range(cur_traj.shape[1]):
                for k in range(len(factors)):
                    if cur_traj[1,j] <= cutoffs[k]:
                        cur_traj[1,j] /= (factors[k]*4.5*in_conv**2)*1000
                        break
                    if k == len(factors)-1:
                        cur_traj[1,j] /= (factors[-1]*4.5*in_conv**2)*1000
            trajs.append(cur_traj)
        return trajs
    
    def optimize_day(trajs, light_conds):
        day_range = [-1,0,1,2,3,4,5,6,7,8,9]
        print('---------------- Data Loading ----------------')

        for i, traj in tqdm(enumerate(trajs), total=len(trajs)):
            res_ = []
            new_traj = [traj[0][:5],traj[1][:5]]
            for day in day_range:
                traj_ = [[np.array([t-day for t in new_traj[0]]), new_traj[1]]]
                mini = Minimizer([4e-7,22.1,0.5], [[0.3e-6, .5e-6],[v*.9, v*1.1], [0.425, 0.575]],traj_, traj_,[light_conds[i],light_conds[i]], spec='None',repress_print=True)
                base, _ = mini.testset_eval(prepare_plot=False)
                res_.append(base)
            best_day_idx = min(enumerate(res_), key=lambda x: x[1])[0]
            trajs[i] = [[t-day_range[best_day_idx] for t in traj[0]], traj[1]]
        return trajs

    factors = [3.5,2.5,3.0,3.5,4.0,4.5,5]
    cutoffs = [3,5,8,11,16,22,26]
    light_conds, raw_trajs = traj_analysis()

    ### Poor Growth Removal (duct-tape removal)
    for i in range(33,39):
        raw_trajs[i] = [raw_trajs[i][0][:-3],raw_trajs[i][1][:-3]]
    raw_trajs[4] = [raw_trajs[4][0][:-3],raw_trajs[4][1][:-3]]

    trajs = pull_wgt_per_area(raw_trajs,factors,cutoffs)
    trajs = optimize_day(trajs, light_conds)

    # mini = Minimizer([4.0e-7,22.1,0.5], bounds,[],trajs,light_conds, spec="None",repress_print=False, huber=False)
    # _, _ = mini.testset_eval(prepare_plot=True)
    # plt.show()

    random.seed(seed)
    indices = list(range(len(trajs)))

    # # Shuffle the indices
    random.shuffle(indices)

    # # Filtering without Replacement
    train_trajs = []
    test_trajs = []
    train_ind = []
    test_ind = []
    filt_light = []
    for i, idx in enumerate(indices):
        if idx not in filter:
            if i < int(train_pct*len(trajs)):
                train_trajs.append(trajs[idx])
                train_ind.append(idx)
            else:
                test_trajs.append(trajs[idx])
                test_ind.append(idx)
            filt_light.append(light_conds[idx])
    light_conds = filt_light[:]
    return train_trajs, test_trajs, light_conds

def K_means_split(batch_size, trajs, light):
    tr_val = []
    N = len(trajs)//batch_size
    for i in range(N):
        if i != (N-1):
            valid = trajs[i*batch_size:(i+1)*batch_size]
            train = trajs[:i*batch_size]+trajs[(i+1)*batch_size:]
            light_tmp = light[:i*batch_size]+light[(i+1)*batch_size:len(trajs)]+light[i*batch_size:(i+1)*batch_size]
        else:
            valid = trajs[i*batch_size:]
            train = trajs[:i*batch_size]
            light_tmp = light[:i*batch_size]+light[i*batch_size:len(trajs)]
        tr_val.append([train,valid, light_tmp])
    return tr_val

if __name__ == '__main__':
    k = 0.4e-6
    v = 22.1
    a = .5
    bounds = [[0.175e-6, .475e-6],[v*.5, v*1.5], [0.325, 0.675]]
    types = ['K','V','A','V_A','VA','KV','KVA','K_V','K_V_A']

    with open("/Users/morganmayborne/Downloads/results_new.csv", mode='r') as f:
        reader = csv.reader(f)
        optim = [item for item in list(reader)]
        optim = optim[8:116]+optim[122:302]
    types = [item[0] for item in optim]
    params = [item[-1] for item in optim]
    print(len(types))
    
    new_params = []
    for p in params:
        param_list = []
        cur_param = ''
        for char in p:
            if char != '[':
                if char == ',' or char == ']':
                    param_list.append(float(cur_param))
                    cur_param = ''
                else:
                    cur_param += char
        
        new_params.append(param_list)

    #### Parameter Set Averages ####
    paramsets = np.zeros((8,27))
    for i in range(len(new_params)):
        paramsets[i//36, i%9] = (paramsets[i//36, i%9]*((i%36)//9)+new_params[i][0])/((i%36)//9+1)
        paramsets[i//36, 9+i%9] = (paramsets[i//36, 9+i%9]*((i%36)//9)+new_params[i][1])/((i%36)//9+1)
        paramsets[i//36, 18+i%9] = (paramsets[i//36, 18+i%9]*((i%36)//9)+new_params[i][2])/((i%36)//9+1)

    # Cumulative Averages
    paramsets = np.mean(paramsets,axis=0)
    [print(types[i], paramsets[i], paramsets[9+i], paramsets[18+i]) for i in range(9)]

    # #### Evaluation ####
    res_ = []
    tr, te, light = create_train_test(0, filter=[1])
    mini = Minimizer([k,v,a], bounds,[], tr+te,light, spec='None',repress_print=True, huber=False)
    base, std_base = mini.testset_eval(prepare_plot=False)
    for p in tqdm(new_params[:]):
        mini = Minimizer(p, bounds,[], tr+te,light, spec='None',repress_print=True, huber=False)
        res, std_res = mini.testset_eval(prepare_plot=False)
        res_.append([p, (res-base)/base, res, base, std_res, std_base])

    #### Averaging Results ####
    averages = np.zeros((8,18*2))
    for i in range(len(res_)):
        averages[i//36,i%9] = (averages[i//36, i%9]*((i%36)//9)+res_[i][2])/((i%36)//9+1)
        averages[i//36,9+i%9] = (averages[i//36, 9+i%9]*((i%36)//9)+res_[i][4])/((i%36)//9+1)
        averages[i//36,18+i%9] = (averages[i//36, 18+i%9]*((i%36)//9)+res_[i][3])/((i%36)//9+1)
        averages[i//36,27+i%9] = (averages[i//36, 27+i%9]*((i%36)//9)+res_[i][5])/((i%36)//9+1)

    # Cumulative Averages
    averages = np.mean(averages,axis=0)
    [print(types[i], averages[i], averages[9+i],averages[i+18], averages[27+i]) for i in range(9)]

    # #### Best / Median Results ####
    res_ = sorted(res_, key=lambda x: x[2])
    print('Best:',res_[0])
    print('Median:',res_[len(res_)//2])
    residuals = [r[2] for r in res_]+[base]
    stdevs = [r[4] for r in res_]+[std_base]
    print('Base Values', base, std_base)

    plt.figure(100)
    r_histogram = plt.hist(residuals, bins=np.arange(base-.2*base,base+.2*base,.2*base/10), color='b')
    plt.title("Parameter Set Value - Histogram") 
    plt.ylabel("Count")
    plt.xlabel("Mean Residual / Measurement (kg/m2)")

    plt.figure(101)
    r_histogram = plt.hist(stdevs, bins=np.arange(std_base-.4*std_base,std_base+.4*std_base,.4*std_base/10),color='r')
    plt.title("Parameter Set Value Std. Deviation - Histogram") 
    plt.ylabel("Count")
    plt.xlabel("Standard Deviation of Residual / Measurement (kg/m2)")
    plt.show()

    #### Testing Certain Trajectories
    # trajs = K_means_split(10, tr+te, light)
    # tr = trajs[0][0]; te = trajs[0][1]; light = trajs[0][2]
    # optimals = [[2.438e-7,22.1,0.5],[4e-7,33.15,0.517],[4e-7,22.1,0.518],[4e-7,33.15,0.518],[4e-7,11.91,0.545],[1.97e-7,26.34,0.5],[2.452e-7,23.53,0.511],[2.43e-7,33.15,0.5],[2.43e-7,33.15,0.504]]
    # for i in range(9):
    #     mini = Minimizer(optimals[i], bounds,[],tr+te,light, spec='None',repress_print=False, huber=False)
    #     base, std_base = mini.testset_eval(prepare_plot=False)
        # mini.optimize_loop()
        # res, std_res = mini.testset_eval(prepare_plot=True)
        # guess = mini.get_final_guess()
        # plt.show()

    # res_total = []
    # for seed in range(10):
    #     print('Seed',seed)
    #     tr, te, light = create_train_test(seed, filter=[1,39,49])
    #     trajs = K_means_split(10, tr+te, light)
    #     for i,traj in enumerate(trajs):
    #         print('Fold',i)
    #         tr = traj[0]; te = traj[1]; light = traj[2]
    #         mini = Minimizer([k,v,a], bounds,tr, te,light, spec='None',repress_print=True, huber=True)
    #         base, std_base = mini.testset_eval(prepare_plot=False)
    #         for t in types:
    #             mini = Minimizer([k,v,a], bounds,tr, te,light, spec=t,repress_print=True, huber=True)
    #             mini.optimize_loop()
    #             res, std_res = mini.testset_eval(prepare_plot=False)
    #             guess = mini.get_final_guess()
    #             print('type:', t,base, std_base, res, std_res, guess)
    #             res_total.append([guess, (res-base)/base, res, base])

    # # [print(r) for r in sorted(res_total, key=lambda x: x[1])]

    # tr, te, light = create_train_test(7, filter=[1,39,49])
    # mini = Minimizer([k,v,a], bounds,tr,te,light, spec='K',repress_print=False, huber=False)
    # base, std_base = mini.testset_eval(prepare_plot=True)
    # mini.optimize_loop()
    # res, std_res = mini.testset_eval(prepare_plot=True)
    # guess = mini.get_final_guess()
    # print(base, std_base, res, std_res, guess)
    # plt.show()
    