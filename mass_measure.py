#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 13:51:08 2024

@author: morganmayborne
"""
import csv, time, random
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from nicolet_model_base import basic_test_w_lighting, residual_function_kva, residual_function_one_param, residual_function_two_params
from generate_trajs import traj_analysis
from scipy.optimize import minimize
import seaborn as sns
import pandas as pd
import scipy.stats as st

plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pacoty.mplstyle')

class Minimizer():
    def __init__(self,init_guess, bounds,train_trajs, test_trajs,lighting, spec='K', huber=False):
        self.init_guess = np.array(init_guess)
        self.bounds = np.array(bounds)
        self.type = spec
        self.huber = huber
        self.huber_delta = 1.0
        self.param_names = ['Maintanence Respiration (K)', 'Growth Rate Coefficient (V)', 'Leaf Area Closure (A)']
        self.train_trajs = train_trajs
        self.test_trajs = test_trajs
        self.light_conds = lighting
        self.trajs = self.train_trajs+self.test_trajs


        self.guess = []
        self.guess_param_setup()

        self.dt = 24

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
            # print([x,self.res_func_s(x)])
            self.history.append([x,self.res_func_s(x)])

        if not self.species:
            return None

        init_guess = np.array(self.init_guess[self.incl==1])
        bounds = self.bounds[self.incl==1,:]

        self.history.append([init_guess,self.res_func_s(init_guess)])
        result = minimize(self.res_func_s, init_guess, method='Powell', bounds=bounds, callback=callback, tol=1e-9)
        
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
        print('------------',self.type, 'Species Optimization Initialized ------------')
        result_k = self.spec_min(plt_history=plt_history)
        print('Initial Guess:',self.init_guess)
        if self.param_order:
            for param in self.param_order:
                if result_k != None:
                    print('Minimized Parameters: ',result_k.x)  # Optimal values
                    print('Minimized Function: ',result_k.fun)  # Minimum function value

                self.change_spec(param)
                result_k = self.spec_min(plt_history=plt_history)
                print('Initial Guess:',self.init_guess)
        
        if result_k != None:
            print('Minimized Parameters: ',result_k.x)  # Optimal values
            print('Minimized Function: ',result_k.fun)  # Minimum function value

        print('------------',self.type, 'Species Optimization Complete ------------')
        residual = 0
        self.err_by_day = []  #### identify the day and error amount for error bar plotting
        self.true_ = []       #### find ground truth values for average
        self.sim = []         #### find simulation values for average

        start = 0
        end = len(self.train_trajs)
        
        print('------------',self.type, 'Plant Optimization Initialized ------------')
        for i in tqdm(range(start, end)):
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
        print('------------ Parameter Evaluation Initialized ------------')

            
        for i in tqdm(range(len(self.train_trajs), len(self.trajs))):
            traj = self.trajs[i]
            lighting = self.light_conds[i]
            _, tmp_residuals = residual_function_kva(self.guess[i][:3], [[traj], [lighting], 1, 0, False])
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
                plt.figure(i)
                plt.plot(np.array(self.trajs[i][0],dtype=np.float64),np.array(self.trajs[i][1],dtype=np.float64),label='M_fm - Ground Truth')
                plt.plot(t_traj[lim[0]:lim[1]]/60/60/24, y_traj[1,lim[0]:lim[1]].T, label="M_fm - Simulation")
                plt.title("Calibrated Simulation")
                plt.xlabel('Time (days)')
                plt.ylabel('Weight (kg m-2)')
                plt.legend()
                plt.grid(True)

        residual = np.array(residuals).mean()
        std_residual = np.std(np.array(residuals))

        ## Violin Plot

        # Flatten the list of errors and create corresponding trajectory labels
        if prepare_plot:
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
            return residual_function_kva(x,self.species_args)

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
                cur_traj[0,j] += -4
                
            trajs.append(cur_traj)
        return trajs
    
    factors = [2.0,2.5,3.0,3.5,4.0,4.5,5.0]
    cutoffs = [2,5,8,11,16,22,26]
    light_conds, raw_trajs = traj_analysis()

    trajs = pull_wgt_per_area(raw_trajs,factors,cutoffs)

    random.seed(seed)
    indices = list(range(len(trajs)))

    # Shuffle the indices
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

def split_train_valid(batch_size, trajs, light):
    tr_val = []
    N = len(tr)//batch_size
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
    exp_seed = 2
    bounds = [[0.3e-6, .5e-6],[v*.75, v*1.5], [0.3, 0.7]]

    param_sets = []
    filters = [[],[49],[49,32,22,39]]
    # types = ['K']
    types = ['K','V','A','KVA','K_V','K_V_A']
    for type_ in types:
        for j,filter in enumerate(filters):
            tr, te, light = create_train_test(exp_seed, filter)
            Kfold_size = int((1/9)*len(tr)) # 80/10/10 splitting train/valid
            train_valid = split_train_valid(Kfold_size, tr, light)
            
            for i in range(len(train_valid)):
                print("Method",type_,",","Filter",j+1,"of",len(filters),",","Fold",i+1,"of",len(train_valid))
                train = train_valid[i][0]
                valid = train_valid[i][1]
                light_tmp = train_valid[i][2]

                mini = Minimizer([k,v,a], bounds,train, valid,light_tmp, spec=type_)
                print("Base Guess")
                base, std_base = mini.testset_eval()
                mini.optimize_loop()
                print("Optimized Guess")
                res, std_res = mini.testset_eval()
                final_guess = mini.get_final_guess()

                mean_diff = res-base
                std_diff = np.sqrt(std_base**2+std_res**2)
                z_score = -mean_diff/std_diff
                confid_score = st.norm.cdf(z_score)

                param_sets.append([final_guess,type_,filter,base,res, std_base, std_res, confid_score])

    param_sets = sorted(param_sets,key= lambda x: x[-1])[::-1]
    [print(params) for params in param_sets]

    params_check = [param_sets[i] for i in range(20)] + [[[k,v,a],'None']]
    res = []
    for p in params_check:
        tr, te, light = create_train_test(exp_seed, filters[1]) ## Chose to do middle filter for all, to compare across filters and the middle filter feels more realistic with what could happen
        mini_train = Minimizer(p[0], bounds,tr, te,light[:len(tr)]+light[:len(tr)], spec=p[1])
        print(mini_train.get_final_guess())
        res_te, _ = mini_train.testset_eval()
        res.append([final_guess,res_te])

    result = sorted(res,key= lambda x: x[-1])[::-1]
    print(result)
        
    # base = 0
    # seeds = [2]
    # filters = [[],[49],[49,32,22,39]]
    # types = ['None','K','V','A','KVA','K_V','K_V_A']
    # param_sets = [[[k,v,a],None,None,None,None]]
    # for seed in seeds:
    #     for filter in filters:
    #         for type in types:
    #             print(seed, filter, type)
    #             mini = Minimizer([k, v, a], bounds, seed, spec=type, filter=filter,train=True)
    #             if type != 'None':
    #                 mini.optimize_loop(prepare_plot=False,plt_history=False)
    #             res = mini.testset_eval(prepare_plot=False)
    #             if type != 'None':
    #                 param_sets.append([mini.get_final_guess(),seed,filter,type, res-base])
    #             else:
    #                 base = res
    
    

    ## note that history is being destroyed with K_V[_A]
    ## Get rid of self.type
    ## Have species change happen through a loop of potential types see K_V_A example
    ## Add random seed to test set generation
    ## (De-prioritized) Need some sort of plant level optimization for everything but K, K_V_A