import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import os, csv, random

# removed = ['2024-09-16-11',
#            '2024-09-17-14',
#            '2024-09-20-7',
#            '2024-10-28-10',
#            '2024-09-24-7',
#            '2024-09-24-2',
#            '2024-09-25-6',
#            '2024-09-24-1',
#            '2024-09-17-8',
#            '2024-09-17-0',
#            '2024-09-17-11',
#            '2024-09-16-2',
#            '2024-09-17-9',
#            '2024-10-31-7',
#            '2024-09-24-2',
#            '2024-09-19-12']
# removed = [r+'.npy' for r in removed]
# removed = os.listdir('/Users/morganmayborne/Downloads/CustomI2GROW_Dataset/RGBDImages/')
# [print(r) for r in removed]

removed = []

for r in removed:
    rgbd = np.load('/Users/morganmayborne/Downloads/CustomI2GROW_Dataset/RGBDImages/'+r)

    rgb = rgbd[:,:,:3]
    d = rgbd[:,:,3]
    d = np.clip(d,1,2000)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    ax1.set_title(r+' - RGB')
    ax1.imshow(rgb)

    # cmap = plt.cm.rainbow
    # norm = matplotlib.colors.Normalize(vmin=0, vmax=25000)
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])  # only needed for matplotlib < 3.1
    # fig.colorbar(sm)
    ax2.set_title(r+' - Depth')
    ax2.imshow(d)
    plt.show()

biomass_csv = '/Users/morganmayborne/Downloads/CustomI2GROW_Dataset/Biomass_Info_Ground_Truth.csv'

with open(biomass_csv, mode='r') as f:
    reader = csv.reader(f)
    read_list = list(reader)

biomass = [int(read_list[i][2]) for i in range(1,len(read_list))]
# biomasses = [[],[]]
# for b in biomass:
#     if b > 15:
#         biomasses[1].append(b)
#     else:
#         biomasses[0].append(b)
# random.shuffle(biomasses[0])
# random.shuffle(biomasses[1])
# biomass = biomasses[0][:int(.7*len(biomasses[0]))]+biomasses[1][:int(.95*len(biomasses[1]))]

random.shuffle(biomass)
biomass = biomass[:int(.8*len(biomass))]

distr = np.histogram(biomass, bins=np.arange(0,np.max(biomass)+1,2))
print(distr)

plt.figure(1)
plt.hist(biomass, bins=np.arange(0,30+1,5), color='r')
plt.title("Training Data Distribution") 
plt.ylabel("Count")
plt.xlabel("Biomass (in g)")
plt.show()
