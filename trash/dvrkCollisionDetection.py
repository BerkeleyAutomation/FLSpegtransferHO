from FLSpegtransfer.motion.deprecated.dvrkMotionBridgeP import dvrkMotionBridgeP
import numpy as np
import FLSpegtransfer.utils.CmnUtil as U

# predict & calibrate interpolated points
def cal_interpolate(self, q_cmd, mode):
    assert q_cmd != []
    assert not self.use_history
    if all(np.isclose(q_cmd, self.q_cmd_int_prev)):
        return q_cmd
    else:
        q_cmd = np.array(q_cmd)
        tf = abs((q_cmd - self.q_cmd_int_prev) / self.v_max)
        n = np.ceil(max(tf / self.dt))  # number of points interpolated
        dq = (q_cmd - self.q_cmd_int_prev) / n
        for i in range(int(n)):
            if mode=='predict':
                q_cmd_new = self.predict(self.q_cmd_int_prev + (i+1)*dq)    # predict
            elif mode=='calibrate':
                q_cmd_new = self.step(self.q_cmd_int_prev + (i+1)*dq)    # calibrate
        self.q_cmd_int_prev = q_cmd
        return q_cmd_new    # return the last one



PSM1 = dvrkMotionBridgeP()
pos_st = [0.07, 0.07, -0.13]
rot1 = [0.0, 0.0, 0.0]
q1 = U.euler_to_quaternion(rot1, unit='deg')
jaw1 = [0*np.pi/180.]
PSM1.set_pose(pos1=pos_st, rot1=q1, jaw1=jaw1)

pos_ed = [0.03, 0.03, -0.13]
while True:
    PSM1.set_pose(pos1=pos_ed, rot1=q1, jaw1=jaw1)