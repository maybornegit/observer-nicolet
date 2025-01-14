import numpy as np

traj_errors = np.zeros((50,))

with open("output_tmp_resid.txt") as f:
    while True:
        line = f.readline()
        if not line:
            break
        if line[0] == 'T':
            continue
        
        format_line = [float(i) for i in line.replace('\n','').split(" ")]
        traj_errors[int(format_line[0])] = format_line[1]

print(len(traj_errors[traj_errors>.25]))
bounds = [0.075, 0.25]
buckets = [[],[],[]]
for i, traj in enumerate(traj_errors):
    if traj < bounds[0]:
        buckets[0].append(i)
    elif traj > bounds[1]:
        buckets[2].append(i)
    else:
        buckets[1].append(i)

print('Good Estimate:','('+str(100*len(buckets[0])/len(traj_errors))+'%)', buckets[0])
print('Moderate Estimate:','('+str(100*len(buckets[1])/len(traj_errors))+'%)', buckets[1])
print('Bad Estimate:', '('+str(100*len(buckets[2])/len(traj_errors))+'%)', buckets[2])