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
from nicolet_model_base import basic_test_w_lighting, residual_function_va, residual_function_kva, residual_function_one_param, residual_function_two_params
from generate_trajs import traj_analysis
from scipy.optimize import minimize
import seaborn as sns
import pandas as pd

plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pacoty.mplstyle')

class Minimizer():
    def __init__(self,init_guess, bounds,seed, thres_ransac=0.025,n_ransac=5, spec='K', train=True, huber=True):
        self.init_guess = np.array(init_guess)
        self.bounds = np.array(bounds)
        self.train = train
        self.type = spec
        self.huber = huber
        self.huber_delta = 1.0
        self.thres_ransac = thres_ransac
        self.n_ransac = n_ransac
        self.param_names = ['Maintanence Respiration (K)', 'Growth Rate Coefficient (V)', 'Leaf Area Closure (A)']

        self.factors = [2.0,2.5,3.0,3.5,4.0,4.5,5.0]
        self.cutoffs = [2,5,8,11,16,22,26]
        self.light_conds, self.raw_trajs = traj_analysis()

        self.trajs = []
        
        self.pull_wgt_per_area()

        random.seed(seed)
        self.indices = list(range(len(self.trajs)))

        # Shuffle the indices
        random.shuffle(self.indices)
        self.trajs = [self.trajs[i] for i in self.indices]
        self.light_conds = [self.light_conds[i] for i in self.indices]

        self.train_trajs = self.trajs[:int(.8*len(self.trajs))]
        self.test_trajs = self.trajs[int(.8*len(self.trajs)):]

        self.guess = []
        self.guess_param_setup()

        self.dt = 24

        self.err_by_day = []  #### identify the day and error amount for error bar plotting
        self.true_ = []       #### find ground truth values for average
        self.sim = []         #### find simulation values for average

        self.history = []

        self.inliers = [1,4,7,9,11,12,14,16,17,20,21,23,24,26,27,28,29,30,31,35,38,40,41,43,44,45,46,47]
        self.midliers = [2,3,5,6,10,13,18,19,25,33,34,36,37,42,48]
        self.outliers = [0,8,15,22,32,39,49]

        self.species = 1
        self.plant = 1
        self.param_order = [] # empty, except for K_V + K_V_A

        if not self.species:
            return None
        trajs = self.trajs[:]
        if self.train:
            trajs = self.train_trajs

        if spec == 'K':
            self.incl = np.array([1,0,0])
            self.spec_args = [trajs,self.light_conds,self.dt, self.huber_delta, self.huber, spec]
            self.plant = 0
        elif spec == 'V':
            self.incl = np.array([0,1,0])
            self.spec_args = [trajs,self.light_conds,self.dt, self.huber_delta, self.huber, spec]
            self.plant = 0
        elif spec == 'A':
            self.incl = np.array([0,0,1])
            self.spec_args = [trajs,self.light_conds,self.dt, self.huber_delta, self.huber, spec]
            self.plant = 0
        elif spec == 'KV':
            self.incl = np.array([1,1,0])
            self.spec_args = [trajs,self.light_conds,self.dt, self.huber_delta, self.huber, spec]
            self.plant = 0
        elif spec == 'VA':
            self.incl = np.array([0,1,1])
            self.spec_args = [trajs,self.light_conds,self.dt, self.huber_delta, self.huber, spec]
            self.plant = 0
        elif spec == 'KA':
            self.incl = np.array([1,0,1])
            self.spec_args = [trajs,self.light_conds,self.dt, self.huber_delta, self.huber, spec]
            self.plant = 0
        elif spec == 'KVA':
            self.incl = np.array([1,1,1])
            self.spec_args = [trajs,self.light_conds,self.dt, self.huber_delta, self.huber, spec]
            self.plant = 0
        elif spec == 'K_V_A':
            self.incl = np.array([1,0,0])
            self.spec_args = [trajs,self.light_conds,self.dt, self.huber_delta, self.huber, 'K']
            self.param_order = ['V','A'] # start with k, then do these
        elif spec == 'K_V':
            self.incl = np.array([1,0,0])
            self.spec_args = [trajs,self.light_conds,self.dt, self.huber_delta, self.huber, 'K']
            self.param_order = ['V'] # start with k, then do these
        elif spec == 'None':
            self.species = 0
            self.plant = 0

        if spec != 'None':
            for guess in self.init_guess[self.incl==0]:
                self.spec_args.append(guess)
        
        # self.randomize_trajs()

    def pull_wgt_per_area(self):##### Find trajectories divided by area
        in_conv = 0.0254
        for i in range(len(self.raw_trajs)):
            cur_traj = np.array(self.raw_trajs[i],dtype=np.float64)
            for j in range(cur_traj.shape[1]):
                for k in range(len(self.factors)):
                    if cur_traj[1,j] <= self.cutoffs[k]:
                        cur_traj[1,j] /= (self.factors[k]*4.5*in_conv**2)*1000
                        break
                    if k == len(self.factors)-1:
                        cur_traj[1,j] /= (self.factors[-1]*4.5*in_conv**2)*1000
                cur_traj[0,j] += -4
                
            self.trajs.append(cur_traj)
        
    def light_reading(self, filename):
        with open(filename,'r') as f:
            reader = csv.reader(f)
            read_list = list(reader)
            self.light_conds = read_list[0]
        self.light_conds = np.array(self.light_conds, dtype=np.float16)
        return None
    
    def guess_param_setup(self):
        for _ in range(len(self.trajs)):
            self.guess.append([self.init_guess[0],self.init_guess[1],self.init_guess[2],1e6])
    
    def spec_min(self, ransac, plt_history=False):
        def callback(x):
            # print([x,self.res_func_s(x)])
            self.history.append([x,self.res_func_s(x)])

        if not self.species:
            return None

        init_guess = np.array(self.init_guess[self.incl==1])
        bounds = self.bounds[self.incl==1,:]

        if not ransac:
            self.history.append([init_guess,self.res_func_s(init_guess)])
            result = minimize(self.res_func_s, init_guess, method='Nelder-Mead', bounds=bounds, callback=callback)
        else:
            best_result = None
            best_inliers = 0
            best_indices = []
            for _ in tqdm(range(20)):
                rnd_indices = self.random_train_set()
                self.history.append([init_guess,self.res_func_s(init_guess)])
                result = minimize(self.res_func_s, init_guess, method='Nelder-Mead', bounds=bounds, callback=callback)

                for i in range(len(self.guess)):
                    count_incl = 0
                    for j in range(4):
                        if j == 3:
                            break
                        if self.incl[j]:
                            self.guess[i][j] = result.x[count_incl]
                            count_incl += 1

                self.spec_args[0] = self.train_trajs
                self.spec_args[1] = self.light_conds
                residual = []
                inliers = 0
                for i in range(0, int(.8*len(self.trajs))):
                    traj = self.trajs[i]
                    lighting = self.light_conds[i]
                    residual.append(residual_function_kva(self.guess[i][:3], [[traj], [lighting], 1, 0, False]))
                    if residual[-1] < self.thres_ransac:
                        inliers += 1
                
                if inliers > best_inliers:
                    best_inliers = inliers
                    best_result = result
                    best_indices = rnd_indices
            result = best_result
        print('Used Training Indices: ',[self.indices[i] for i in best_indices])
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
    
    def optimize_loop(self,ransac=False,prepare_plot=False, plt_history=False):
        print('------------',self.type, 'Species Optimization Initialized ------------')
        result_k = self.spec_min(ransac, plt_history=plt_history)
        print('Initial Guess:',self.init_guess)
        if self.param_order:
            for param in self.param_order:
                if result_k != None:
                    print('Minimized Parameters: ',result_k.x)  # Optimal values
                    print('Minimized Function: ',result_k.fun)  # Minimum function value

                self.change_spec(param)
                result_k = self.spec_min(ransac,plt_history=plt_history)
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
        end = len(self.trajs)
        if self.train:
            end = int(.8*len(self.trajs))
        
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

    def testset_eval(self, prepare_plot=True):
        residual = 0
        inl_res = []
        midl_res = []
        outl_res = []
        self.err_by_day = []  #### identify the day and error amount for error bar plotting
        self.true_ = []       #### find ground truth values for average
        self.sim = []         #### find simulation values for average
        print('------------ Parameter Evaluation Initialized ------------')
            
        for i in tqdm(range(int(.8*(len(self.trajs))), len(self.trajs))):
            traj = self.trajs[i]
            lighting = self.light_conds[i]
            tmp_r = residual
            residual += residual_function_kva(self.guess[i][:3], [[traj], [lighting], 1, 0, False])
            # residual += residual_function_kva([4.75e-7, 22.1,0.5], [[traj], [lighting], 1, 0, False])

            print(self.indices[i],residual-tmp_r)
            if self.indices[i] in self.inliers:
                inl_res.append(residual-tmp_r)
            elif self.indices[i] in self.midliers:
                midl_res.append(residual-tmp_r)
            else:
                outl_res.append(residual-tmp_r)

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
        print(inl_res, midl_res, outl_res)
        print("Residual per Traj:", residual / len(self.test_trajs))
        print("Residual per Inlier:", np.array(inl_res).mean())
        print("Residual per Midlier:",np.array(midl_res).mean())
        print("Residual per Outlier:", np.array(outl_res).mean())
        print('------------ Parameter Evaluation Complete ------------\n')

        # with open("ransac_results_1.txt",'a') as f:
        #     if not self.species:
        #         f.write('Base Inlier: '+' '+str(np.array(inl_res).mean())+'\n')
        #         f.write('Base Midlier: '+' '+str(np.array(midl_res).mean())+'\n')
        #         f.write('Base Outlier: '+' '+str(np.array(outl_res).mean())+'\n')
        #     else:
        #         f.write(str(self.n_ransac)+' '+str(self.thres_ransac)+' '+'Inlier: '+' '+str(np.array(inl_res).mean())+'\n')
        #         f.write(str(self.n_ransac)+' '+str(self.thres_ransac)+' '+'Inlier: '+' '+str(np.array(midl_res).mean())+'\n')
        #         f.write(str(self.n_ransac)+' '+str(self.thres_ransac)+' '+'Inlier: '+' '+str(np.array(outl_res).mean())+'\n')
        return residual / len(self.test_trajs)
 
    def res_func_s(self, x):
        type_res = self.spec_args[5]

        if type_res == 'K':
            return residual_function_one_param(x,self.spec_args)
        elif type_res == 'V':
            return residual_function_one_param(x,self.spec_args)
        elif type_res == 'A':
            return residual_function_one_param(x,self.spec_args)
        elif type_res == 'KV':
            return residual_function_two_params(x,self.spec_args)
        elif type_res == 'VA':
            return residual_function_two_params(x,self.spec_args)
        elif type_res == 'KA':
            return residual_function_two_params(x,self.spec_args)
        elif type_res == 'KVA':
            return residual_function_kva(x,self.spec_args)

    def res_func_p(self, x, args):
        if self.type == 'K':
            # return residual_function_va(x,args)
            raise NotImplementedError
        else:
            raise NotImplementedError

    def change_spec(self, new_type):
        trajs = self.trajs[:]
        if self.train:
            trajs = self.train_trajs

        if new_type == 'K':
            self.incl = np.array([1,0,0])
            self.spec_args = [trajs,self.light_conds,self.dt, self.huber_delta, self.huber, new_type]
        elif new_type == 'V':
            self.incl = np.array([0,1,0])
            self.spec_args = [trajs,self.light_conds,self.dt, self.huber_delta, self.huber, new_type]
            self.plant = 0
        elif new_type == 'A':
            self.incl = np.array([0,0,1])
            self.spec_args = [trajs,self.light_conds,self.dt, self.huber_delta, self.huber, new_type]
            self.plant = 0
        elif new_type == 'KV':
            self.incl = np.array([1,1,0])
            self.spec_args = [trajs,self.light_conds,self.dt, self.huber_delta, self.huber, new_type]
            self.plant = 0
        elif new_type == 'VA':
            self.incl = np.array([0,1,1])
            self.spec_args = [trajs,self.light_conds,self.dt, self.huber_delta, self.huber, new_type]
            self.plant = 0
        elif new_type == 'KA':
            self.incl = np.array([1,0,1])
            self.spec_args = [trajs,self.light_conds,self.dt, self.huber_delta, self.huber, new_type]
            self.plant = 0
        
        for guess in self.init_guess[self.incl==0]:
            self.spec_args.append(guess)

        self.init_guess = np.array([self.guess[0][0],self.guess[0][1], self.guess[0][2]])
            
        return None

    def random_train_set(self):
        random.seed(time.time())
        indices = list(range(len(self.train_trajs)))

        # Shuffle the indices
        random.shuffle(indices)
        self.train_trajs_rnd = [self.train_trajs[i] for i in indices[:self.n_ransac]]
        self.light_conds_rnd = [self.light_conds[i] for i in indices[:self.n_ransac]]

        self.history = []

        self.spec_args[0] = self.train_trajs_rnd
        self.spec_args[1] = self.light_conds_rnd
        return indices[:self.n_ransac]

if __name__ == '__main__':
    k = 0.4e-6
    v = 22.1
    a = .5
    bounds = [[0.3e-6, .5e-6],[v*.75, v*1.5], [0.3, 0.7]]

    # with open("ransac_results_1.txt",'w') as f:
    #     f.write("Ransac Results"+'\n')

    mini_base = Minimizer([k, v, a], bounds, 10, spec='None', train=True, huber=False)
    mini_base.testset_eval(prepare_plot=True)
    plt.show()
    # hyper_p = [(5,0.075),(5,0.09),(5,0.25),(6,0.075),(6,0.09),(6,0.25),(7,0.075),(7,0.09),(7,0.25),(8,0.075),(8,0.09),(8,0.25),(9,0.075),(9,0.09),(9,0.25),(10,0.075),(10,0.09),(10,0.25)]

    ## Sweep Across All Types
    # for i in range(18):
    #     n, threshold = hyper_p[i]
    #     # mini_base = Minimizer([k, v, a], bounds, seed, spec='None', train=True, huber=False)
    #     mini_k = Minimizer([k, v, a], bounds, 10,thres_ransac=threshold, n_ransac=n,spec='K', train=True, huber=False)
    #     # mini_v = Minimizer([k, v, a], bounds,seed, spec='V', train=True, huber=False)
    #     # mini_a = Minimizer([k, v, a], bounds,seed, spec='A', train=True, huber=False)
    #     # mini_kv = Minimizer([k, v, a], bounds,seed, spec='KV', train=True, huber=False)
    #     # mini_kva = Minimizer([k, v, a], bounds,seed, spec='KVA', train=True, huber=False)
    #     # mini_k_v = Minimizer([k, v, a], bounds,seed, spec='K_V', train=True, huber=False)
    #     # mini_k_v_a = Minimizer([k, v, a], bounds,seed, spec='K_V_A', train=True, huber=False)
    #     # ablation_minis = [mini_base, mini_k, mini_v, mini_a, mini_kv, mini_kva, mini_k_v, mini_k_v_a]
    #     ablation_minis = [mini_k]

    #     for mini in ablation_minis:
    #         residual = mini.optimize_loop(ransac=True, prepare_plot=False,plt_history=False)
    #         # mini.avg_and_error_plot(avg=True, error=True)
    #         mini.testset_eval(prepare_plot=False)
    #         # plt.show()

    ## K Optimization
    # mini_k = Minimizer([k, v, a], bounds, 10,thres_ransac=0.25, n_ransac=5,spec='K', train=True, huber=False)
    # residual = mini_k.optimize_loop(ransac=True, prepare_plot=False,plt_history=False)
    # mini_k.testset_eval(prepare_plot=True)
    # plt.show()

    ## note that history is being destroyed with K_V[_A]
    ## Get rid of self.type
    ## Have species change happen through a loop of potential types see K_V_A example
    ## Add random seed to test set generation
    ## (De-prioritized) Need some sort of plant level optimization for everything but K, K_V_A