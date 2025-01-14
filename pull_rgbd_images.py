import pyrealsense2 as rs
import numpy as np
import time, os
from datetime import datetime as dt

today = dt.today()
n = 27
my_dir = os.path.expanduser(("~/Downloads/biomass_data/"+today.strftime('%Y-%m-%d')+'_176').replace('-',''))
if not os.path.exists(my_dir):
   os.mkdir(my_dir)

pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

for i in range(n):
    frames = pipeline.wait_for_frames()

    # Wait for a coherent pair of frames: depth and color
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        raise NotImplementedError

    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data()).reshape((480, 640, 1))
    color_image = np.asanyarray(color_frame.get_data())

    rgbd_frame = np.concatenate((color_image, depth_image), axis=-1)

    np.save(my_dir+"/"+today.strftime('%Y-%m-%d')+"-"+str(i)+".npy", rgbd_frame)
    if i != (n-1):
        input("Next: ")
        time.sleep(2)

pipeline.stop()
