from FLSpegtransfer.motion.dvrkArm import dvrkArm
from FLSpegtransfer.path import *
import time
import numpy as np

p1 = dvrkArm('/PSM1')
joint1_record = []
try:
    time_curr = 0.0
    time_st = time.perf_counter()
    while time_curr < 50:
        time_curr = time.perf_counter() - time_st
        joint1 = p1.get_current_joint(wait_callback=True)
        time.sleep(0.005)
        joint1_record.append(joint1)
        print(joint1, time_curr)
finally:
    np.save(root+"new_traj", joint1_record)
    print("Data is successfully saved")