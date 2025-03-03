import numpy as np
import matplotlib.pyplot as plt
import os

### Showing Pics
# removed = []
# # removed += ["2024-10-17-13"]+["2024-10-22-10"]+["2024-10-25-8"]+["2024-10-29-10"] # Bad
# # removed += ["2024-10-09-4"]+["2024-10-14-10"]+["2024-10-17-10"]+["2024-10-21-8"] # Good
# removed += ["2024-09-"+str(i)+"-9" for i in [10,11,16]]+["2024-09-19-13"]  # Bad
# removed += ["2024-10-"+str(i)+"-14" for i in [16,17]]+["2024-10-22-11"]+["2024-10-25-11"] # Good
# removed = [r+'.npy' for r in removed]
# # removed = os.listdir('/Users/morganmayborne/Downloads/CustomI2GROW_Dataset/RGBDImages/')
# # [print(r) for r in removed]

# for r in removed:
#     rgbd = np.load('/Users/morganmayborne/Downloads/CustomI2GROW_Dataset/RGBDImages/'+r)

#     rgb = rgbd[:,:,:3]
#     d = rgbd[:,:,3]
#     d = np.clip(d,1,2000)
#     plt.figure(1)
#     plt.imshow(rgb)
#     plt.title(r)
#     plt.show()


############################################
### Eff Net Estimates

import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
from test_functions_20250303 import load_net, get_wgt_estimate, labelled_image, prep_image

coreDir = os.path.expanduser("~/Downloads/MSR/CustomI2GROW_Dataset/")
trueDir = "/Biomass_Info_Ground_Truth.csv"
images = os.listdir('/Users/morganmayborne/Downloads/CustomI2GROW_Dataset/RGBDImages/')
images.sort()

net = load_net("/Users/morganmayborne/Downloads/2025_0203_test-1-effnet-c.pt")
net.eval()

rgb_folder = '/Users/morganmayborne/Downloads/CustomI2GROW_Dataset/RGBImages/'
rgbd_folder = '/Users/morganmayborne/Downloads/CustomI2GROW_Dataset/RGBDImages/'

for i in tqdm(range(5)):
    rgbd = np.load(rgbd_folder+images[i])

    trueData = pd.read_csv(coreDir + trueDir)
    true_mass = float(trueData[trueData['Data ID'] == images[i].replace('.npy', '')]['Fresh Biomass'].iloc[0])
    if true_mass > 30:
        continue

    rgbd = prep_image(rgbd_folder+images[i])
    wgt = get_wgt_estimate(net, rgbd)
    full_image = labelled_image(rgb_folder+images[i], wgt)
    cv2.imshow('Plant Estimation', full_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()