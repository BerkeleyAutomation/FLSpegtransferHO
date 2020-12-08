from FLSpegtransfer.motion.dvrkArm import dvrkArm
from FLSpegtransfer.training.dvrkCurrEstDNN import dvrkCurrEstDNN
from FLSpegtransfer.training.dvrkHystCompDNN import dvrkHystCompDNN
from FLSpegtransfer.motion.dvrkKinematics import dvrkKinematics
from FLSpegtransfer.utils.Filter.filters import LPF
import FLSpegtransfer.utils.CmnUtil as U
import time
import numpy as np


class dvrkController(dvrkArm):
    def __init__(self, arm_name, comp_hysteresis=False, stop_collision=False):
        super(dvrkController, self).__init__(arm_name=arm_name)

        # Members
        self.stop_collision = stop_collision
        self.comp_hysteresis = comp_hysteresis
        self.curr_threshold_low = np.array([[0.1, 0.1, 0.18], [0.1, 0.1, 0.18]])
        self.curr_threshold_high = self.curr_threshold_low*1.5
        self._curr_threshold = self.curr_threshold_low
        self.filter = LPF(fc=3, fs=100, order=2, nb_axis=3)

        # Load models
        if stop_collision:
            self.force_est = dvrkCurrEstDNN(history=50, nb_ensemble=10, nb_axis=3)  # torque prediction model
            for i in range(100):     # model start-up by padding dummy histories
                self.detect_collision()
        if comp_hysteresis:
            self.hyst_comp = dvrkHystCompDNN(history=6, arm_name=arm_name)  # hysteresis compensation model
        self.dvrk_model = dvrkKinematics()

    @property
    def curr_threshold(self):
        return self._curr_threshold

    @curr_threshold.setter
    def curr_threshold(self, value):
        self._curr_threshold = value
        # print("curr_threshold updated=", self._curr_threshold)

    def set_pose(self, pos=None, rot=None, use_ik=True, wait_callback=True):
        if pos is None:
            pos = []
        if rot is None:
            rot = []
        if self.comp_hysteresis:
            joint = self.dvrk_model.pose_to_joint(pos, rot)
            return self.set_joint(joint=joint, wait_callback=wait_callback, use_interpolation=False)
        else:
            return super().set_pose(pos=pos, rot=rot, wait_callback=wait_callback)
            # return self.set_pose_interpolate(pos1=pos1, rot1=rot1, pos2=pos2, rot2=rot2, method='cubic')

    # specify intermediate points between q0 & qf
    def set_pose_interpolate(self, pos=[], rot1=[], pos2=[], rot2=[], method='cubic'):
        # Define q0 and qf for arm 1
        pos10, rot10 = self.arm1.get_current_pose(wait_callback=True)
        if len(pos1) == 0:
            pos1f = pos10
        else:
            pos1f = pos1

        if len(rot1) == 0:
            rot1f = rot10
        else:
            rot1f = rot1

        rot10 = U.quaternion_to_euler(rot10)
        rot1f = U.quaternion_to_euler(rot1f)
        q10 = np.concatenate((pos10, rot10))
        q1f = np.concatenate((pos1f, rot1f))

        # Define q0 and qf for arm 2
        pos20, rot20 = self.arm2.get_current_pose(wait_callback=True)
        if len(pos2) == 0:
            pos2f = pos20
        else:
            pos2f = pos2

        if len(rot2) == 0:
            rot2f = rot20
        else:
            rot2f = rot2
        rot20 = U.quaternion_to_euler(rot20)
        rot2f = U.quaternion_to_euler(rot2f)
        q20 = np.concatenate((pos20, rot20))
        q2f = np.concatenate((pos2f, rot2f))

        # Define trajectory
        v_max = [0.1, 0.1, 0.1, 1.0, 1.0, 1.0]
        a_max = [0.1, 0.1, 0.1, 1.0, 1.0, 1.0]
        if method == 'cubic':
            time1, traj1 = self.arm1.cubic(q10, q1f, v_max=v_max, a_max=a_max, t_step=0.01)
            time2, traj2 = self.arm2.cubic(q20, q2f, v_max=v_max, a_max=a_max, t_step=0.01)
        elif method == 'LSPB':
            time1, traj1 = self.arm1.LSPB(q10, q1f, v_max=v_max, a_max=a_max, t_step=0.01)
            time2, traj2 = self.arm2.LSPB(q20, q2f, v_max=v_max, a_max=a_max, t_step=0.01)
        else:
            raise IndexError

        # Execute trajectory
        len1 = len(traj1)
        len2 = len(traj2)
        len_max = max(len1, len2)
        traj_add1 = np.repeat([traj1[-1]], len_max-len1, axis=0)
        traj_add2 = np.repeat([traj2[-1]], len_max-len2, axis=0)
        traj1 = np.concatenate((traj1, traj_add1), axis=0)
        traj2 = np.concatenate((traj2, traj_add2), axis=0)
        for i, (q1, q2) in enumerate(zip(traj1, traj2)):
            super().set_pose(pos1=q1[:3], rot1=U.euler_to_quaternion(q1[3:]),
                             pos2=q2[:3], rot2=U.euler_to_quaternion(q2[3:]), wait_callback=False)
            if self.stop_collision:
                if self.detect_collision():
                    # when collide, temporary make the threshold higher to escape in next move.
                    self.curr_threshold = self.curr_threshold_high
                    return False
            time.sleep(0.01)
        # wait until goal reached
        super().set_pose(pos1=traj1[-1][:3], rot1=U.euler_to_quaternion(traj1[-1][3:]),
                         pos2=traj2[-1][:3], rot2=U.euler_to_quaternion(traj2[-1][3:]), wait_callback=True)
        self.curr_threshold = self.curr_threshold_low
        return True

    # only for stack dummy joint hysteresis
    def set_joint_dummy(self, joint=None, wait_callback=True, use_interpolation=False):
        if joint is None:
            joint = []
        if self.comp_hysteresis:  # update new joint values
            if use_interpolation:  # use interpolated input to the hyst_comp model
                self.hyst_comp.cal_interpolate(joint, mode='calibrate')
            else:
                self.hyst_comp.step(joint)

    def set_joint(self, joint=None, wait_callback=True, use_interpolation=False):
        if joint is None:
            joint = []
        if self.comp_hysteresis:    # update new joint values
            if use_interpolation:   # use interpolated input to the hyst_comp model
                joint = self.hyst_comp.cal_interpolate(joint, mode='calibrate')
            else:
                joint = self.hyst_comp.step(joint)
        return super().set_joint(joint=joint, wait_callback=wait_callback)
        # return self.set_joint_interpolate(joint1=joint1, joint2=joint2, method='cubic')

    def set_joint_interpolate(self, joint=[], method='cubic'):
        # Define q0 and qf for arm 1
        q10 = self.arm1.get_current_joint(wait_callback=True)
        if len(joint) == 0:
            q1f = q10
        else:
            q1f = joint

        # Define q0 and qf for arm 2
        q20 = self.arm2.get_current_joint(wait_callback=True)
        if len(joint2) == 0:
            q2f = q20
        else:
            q2f = joint2

        # Define trajectory
        v_max = np.array([0.5, 0.5, 0.2, 2.0, 2.0, 2.0])*3
        a_max = np.array([0.5, 0.5, 0.2, 2.0, 2.0, 2.0])*3
        if method == 'cubic':
            time1, traj1 = self.arm1.cubic(q10, q1f, v_max=v_max, a_max=a_max, t_step=0.01)
            time2, traj2 = self.arm2.cubic(q20, q2f, v_max=v_max, a_max=a_max, t_step=0.01)
        elif method == 'LSPB':
            time1, traj1 = self.arm1.LSPB(q10, q1f, v_max=v_max, a_max=a_max, t_step=0.01)
            time2, traj2 = self.arm2.LSPB(q20, q2f, v_max=v_max, a_max=a_max, t_step=0.01)
        else:
            raise IndexError

        # Execute trajectory
        len1 = len(traj1)
        len2 = len(traj2)
        len_max = max(len1, len2)
        traj_add1 = np.repeat([traj1[-1]], len_max - len1, axis=0)
        traj_add2 = np.repeat([traj2[-1]], len_max - len2, axis=0)
        traj1 = np.concatenate((traj1, traj_add1), axis=0)
        traj2 = np.concatenate((traj2, traj_add2), axis=0)
        for i, (q1, q2) in enumerate(zip(traj1, traj2)):
            super().set_joint(joint1=q1, joint2=q2, wait_callback=False)
            if self.stop_collision:
                if self.detect_collision():
                    # when collide, temporary make the threshold higher to escape in next move.
                    self.curr_threshold = self.curr_threshold_high
                    return False
            time.sleep(0.01)
        super().set_joint(joint1=traj1[-1], joint2=traj2[-1], wait_callback=True)    # wait until goal reached
        self.curr_threshold = self.curr_threshold_low
        return True

    def detect_collision(self):
        # estimate motor current
        joint1 = self.arm1.get_current_joint(wait_callback=False)
        curr1_est = self.force_est.predict(joint1[:3])

        # measure actual current
        curr1_act = self.arm1.get_motor_current(wait_callback=False)
        curr1_act = self.filter.filter([curr1_act[:3]])[0]
        curr_comp = curr1_act - curr1_est

        if self.stop_collision and (abs(curr_comp) > self.curr_threshold[0]).any():
            # print("Measured_current=", curr_comp)
            # print (self.curr_threshold)
            return True
        else:
            return False


if __name__ == "__main__":
    dvrk = dvrkController(comp_hysteresis=False, stop_collision=True)
    pos1 = [0.03, -0.004, -0.150]
    pos2 = [0.06, -0.004, -0.150]
    rot1 = [0.0, 0.0, 0.0]
    q1 = U.euler_to_quaternion(rot1, unit='deg')
    jaw1 = [0*np.pi/180.]
    # perception.set_pose(jaw1=jaw1)

    # Set joint of the end effector
    joint = [0.25, 0.03, 0.15, 0.0, -0.03, -0.26]  # in (rad) or (m)
    jaw = [0 * np.pi / 180.]  # jaw angle in (rad)
    dvrk.set_joint(joint1=joint)
    dvrk.set_jaw(jaw1=jaw)

    while True:
        print ("pos1 start")
        print(dvrk.set_pose(pos1=pos1, rot1=q1))
        dvrk.set_jaw(jaw1=jaw)
        time.sleep(1)
        print("pos2 start")
        print(dvrk.set_pose(pos1=pos2, rot1=q1))
        dvrk.set_jaw(jaw1=jaw)
        time.sleep(1)
