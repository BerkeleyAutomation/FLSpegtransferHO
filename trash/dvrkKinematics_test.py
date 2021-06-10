from FLSpegtransfer.motion.dvrkKinematics import dvrkKinematics
import numpy as np
import FLSpegtransfer.utils.CmnUtil as U

model = dvrkKinematics()

pos = [0.3, 1.03, -0.15]
rot = [0.0, -20.0, -3.0]
quat = U.euler_to_quaternion(rot, 'deg')

import time
st = time.time()
ans = model.pose_to_joint(pos, quat, method='analytic')
ans2 = model.pose_to_joint(pos, quat, method='numerical')

# print (ans)
# print (ans2)

L1 = 0.1
L2 = 0.2
L3 = 0.02
L4 = 0.05
st = time.time()
a = dvrkKinematics.fk(ans, L1, L2, L3, L4)
print (time.time() - st)

st = time.time()
b = dvrkKinematics.fk_orientation(ans)
print (time.time() - st)

print (a)
print (b)