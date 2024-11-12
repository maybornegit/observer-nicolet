import os
from datetime import datetime as dt

n = 20
today = dt.today()
print(today.strftime('%Y-%m-%d'))
dir = "/Users/morganmayborne/Downloads/MSR/test_rgbd_frames/"

for i in range(n):
    os.rename(dir+today.strftime('%Y-%m-%d')+str(i)+".npy",dir+today.strftime('%Y-%m-%d')+"-"+str(i)+".npy")