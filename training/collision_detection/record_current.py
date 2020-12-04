from FLSpegtransfer.motion.dvrkArm import dvrkArm
import time
import numpy as np

p1 = dvrkArm('/PSM1')

joint1 = []
mot_curr1 = []
time_stamp = []
cnt = 0.0
nStart = 0.0
interval_ms = 10
try:
    while cnt < 20.0:
        nEnd = time.time()
        if nEnd - nStart < interval_ms * 0.001:
            pass
        else:
            joint1.append(p1.get_current_joint(wait_callback=False))
            mot_curr1.append(p1.get_motor_current(wait_callback=False))
            time_stamp.append(cnt)
            cnt += 0.01
            nStart = nEnd
            print (cnt)
finally:
    np.save("joint_msd", joint1)
    np.save("joint_des", joint1)
    np.save("mot_curr", mot_curr1)
    print("Data is successfully saved.")