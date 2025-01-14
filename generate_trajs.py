import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def get_traj(file_sheet, mass_csv):
    suc = pd.read_csv(file_sheet)
    suc = suc[~suc["Data ID 2"].isnull()]
    init = suc["Data ID 1"].to_list()
    mass = mass_csv

    successors = []
    traj = 0
    for i in range(len(init)):
        unexplored = 0
        for j in range(len(successors)):
            if init[i] not in successors[j]:
                unexplored += 1
        if unexplored == len(successors):
            cur_suc = init[i]
            successors.append([])
            while True:
                successors[traj].append(cur_suc)
                cur_suc = suc[suc["Data ID 1"] == successors[traj][-1]]["Data ID 2"]
                try:
                    cur_suc = cur_suc.iloc[0]
                except:
                    traj += 1
                    break
    successors = sorted(successors, key=lambda x: len(x))
    adjacent_pics = []
    for i in range(len(successors)):
        for jump in range(1,min(len(successors[i]),4)):
            for j in range(len(successors[i]) - jump):
                day_1 = successors[i][j]
                day_2 = successors[i][j + jump]
                if jump == 1:
                    delta = int(suc[suc["Data ID 1"] == day_1]["Delta"].iloc[0])
                else:
                    try:
                        mass_1 = int(mass[mass["Data ID"] == day_1]["Fresh Biomass"].iloc[0])
                        mass_2 = int(mass[mass["Data ID"] == day_2]["Fresh Biomass"].iloc[0])
                        delta = max(mass_2-mass_1,0)
                    except:
                        break
                new_set = [day_1, day_2, delta]
                date_1 = datetime.strptime(day_1[:10], "%Y-%m-%d")
                date_2 = datetime.strptime(day_2[:10], "%Y-%m-%d")
                new_set.append(int((date_2 - date_1).days))
                adjacent_pics.append(new_set)
    return adjacent_pics, successors

def ground_truth_traj(seq,trueData):
    traj_array_g = [[], []]
    for j in range(len(seq)):
        traj_array_g[1].append(
            float(trueData[trueData['Data ID'] == seq[j].replace('.npy', '')]['Fresh Biomass'].iloc[0]))
        if j == 0:
            traj_array_g[0].append(10)
        else:
            diff = (datetime.strptime(seq[j][:10], "%Y-%m-%d") - datetime.strptime(seq[j - 1][:10],
                                                                                        "%Y-%m-%d")).days
            traj_array_g[0].append(traj_array_g[0][-1] + int(diff))
        if (traj_array_g[0][-1] - 10) >= 14:
            break
    return traj_array_g[0], traj_array_g[1]

def lighting(traj, trueData, light_idx):
    lights = []
    for pt in traj:
        pos = trueData[trueData['Data ID'] == pt.replace('.npy', '')]['Position'].iloc[0]
        light = light_idx[light_idx['Position Index'] == pos]['Lighting Value'].iloc[0]
        lights.append(light)
    return np.mean(np.array(lights))

def traj_analysis():
    coreDir = os.path.expanduser("~/Downloads/MSR/CustomI2GROW_Dataset/")
    trueDir = "/Biomass_Info_Ground_Truth.csv"
    lightDir = "/lighting.csv"
    suc_sheet = coreDir + '/successor_sheet.csv'
    trueData = pd.read_csv(coreDir + trueDir)
    lightidx = pd.read_csv(coreDir + lightDir)

    ### Relevant Trajectory Loading
    data, trajs = get_traj(suc_sheet, trueData)
    test_data_names = ['2024-09-10-0', '2024-09-11-0', '2024-09-12-0', '2024-09-13-0','2024-09-16-0', '2024-09-17-0', '2024-09-18-0', '2024-09-19-0', '2024-09-20-0',
                '2024-09-10-3','2024-09-11-3', '2024-09-12-3', '2024-09-13-3', '2024-09-16-3', '2024-09-17-3', '2024-09-18-3','2024-09-19-7', '2024-09-20-7',
                '2024-09-10-7', '2024-09-11-7', '2024-09-12-7', '2024-09-13-7','2024-09-16-7', '2024-09-17-7', '2024-09-18-7', '2024-09-19-11', '2024-09-20-11', '2023-09-24-4', '2024-09-25-3', '2024-09-26-3', '2024-09-27-3']

    # For finding average lifespans: Remember to comment out ground_truth_traj if line that cuts off at 14 days
    # traj_pred = []
    # for i in range(len(trajs)):
    #     traj_array_g = list(ground_truth_traj(trajs[i],trueData))
    #     traj_pred.append([i, traj_array_g])
    # print([x[1][0][-1]-x[1][0][0] for x in traj_pred])
    # print(np.mean(np.array([x[1][0][-1]-x[1][0][0] for x in traj_pred])))
    trajs_filt = []
    light = []
    for i in range(len(trajs)):
        if trueData[trueData['Data ID'] == trajs[i][0].replace('.npy', '')]['Fresh Biomass'].iloc[0] <= 3 and len(trajs[i]) >= 9:
            trajs_filt.append(trajs[i])
            light.append(lighting(trajs[i],trueData,lightidx))
    trajs = trajs_filt[:]

    ### Filtering Raw Data
    test_data = []
    for i in range(len(data)):
        if data[i][0] in test_data_names or data[i][1] in test_data_names:
            test_data.append(data[i])
    data = test_data[:]

    ### Predictions Mapped to Trajectories
    traj_pred = []
    for i in range(len(trajs)):
        traj_array_g = list(ground_truth_traj(trajs[i],trueData))
        traj_pred.append([i, traj_array_g])
    return light, [t[1] for t in traj_pred]

if __name__ == '__main__':
    light, trajs = traj_analysis()
    print(len(trajs))