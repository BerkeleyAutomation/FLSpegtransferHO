from FLSpegtransferHO.motion.dvrkArm import dvrkArm
from FLSpegtransferHO.training.dvrkCurrEstDNN import dvrkCurrEstDNN
from FLSpegtransferHO.training.dvrkHystCompDNN import dvrkHystCompDNN
from FLSpegtransferHO.motion.dvrkKinematics import dvrkKinematics
import FLSpegtransferHO.motion.dvrkVariables as dvrkVar
from FLSpegtransferHO.utils.Filter.filters import LPF
import FLSpegtransferHO.utils.CmnUtil as U
import time
import numpy as np
from FLSpegtransferHO.path import *


class dvrkController(dvrkArm):
    def __init__(self, arm_name, comp_hysteresis=False, use_interpolation=False, stop_collision=False):
        super().__init__(arm_name=arm_name)

        # Members
        self.stop_collision = stop_collision
        self.comp_hysteresis = comp_hysteresis
        self.use_interpolation = use_interpolation
        self.curr_threshold_low = np.array([[0.1, 0.1, 0.18], [0.1, 0.1, 0.18]])
        self.curr_threshold_high = self.curr_threshold_low*1.5
        self._curr_threshold = self.curr_threshold_low
        self.filter = LPF(fc=3, fs=100, order=2, nb_axis=3)
        self.last_joint_cmd = []

        # Load models
        if self.stop_collision:
            self.force_est = dvrkCurrEstDNN(history=50, nb_ensemble=10, nb_axis=3)  # torque prediction model
            for i in range(100):     # model start-up by padding dummy histories
                self.detect_collision()
        if self.comp_hysteresis:
            self.hyst_comp = dvrkHystCompDNN(history=6, arm_name=arm_name)  # hysteresis compensation model

        q_phy = super().get_current_joint(wait_callback=True)
        self.last_joint_cmd = q_phy
        for i in range(20):
            self.hyst_comp.step(q_phy)

    @property
    def curr_threshold(self):
        return self._curr_threshold

    @curr_threshold.setter
    def curr_threshold(self, value):
        self._curr_threshold = value
        # print("curr_threshold updated=", self._curr_threshold)

    def get_current_pose(self):
        return dvrkKinematics.joint_to_pose(self.last_joint_cmd)

    def get_current_joint(self):
        return self.last_joint_cmd

    def set_pose(self, pos=None, rot=None, use_ik=True, wait_callback=True):
        if pos is None:
            pos = []
        if rot is None:
            rot = []
        if self.comp_hysteresis:
            joint = dvrkKinematics.pose_to_joint(pos, rot)
            return self.set_joint(joint=joint, wait_callback=wait_callback, use_interpolation=False)
        else:
            return super().set_pose(pos=pos, rot=rot, wait_callback=wait_callback)
            # return self.set_pose_interpolate(pos1=pos1, rot1=rot1, pos2=pos2, rot2=rot2, method='cubic')

    def set_pose_interpolate(self, pos=None, rot=None, tf_init=0.5, t_step=0.01):
        pos0, quat0 = dvrkKinematics.joint_to_pose(self.last_joint_cmd)
        if len(pos) == 0:
            posf = pos0
        else:
            posf = pos
        if len(rot) == 0:
            quatf = quat0
        else:
            quatf = rot
        rot0 = U.quaternion_to_euler(quat0)
        rotf = U.quaternion_to_euler(quatf)
        pose0 = np.concatenate((pos0, rot0))
        posef = np.concatenate((posf, rotf))

        # Define trajectory
        if np.allclose(pose0, posef):
            return False
        else:
            _, q_pos = self.cubic_cartesian(pose0, posef, vel_limit=dvrkVar.v_max, acc_limit=dvrkVar.a_max,
                                           tf_init=tf_init, t_step=t_step)
        # Execute trajectory
        for q in q_pos:
            start = time.perf_counter()
            self.set_joint_direct(q)
            end = time.perf_counter()
            delta = 0.01 - (end - start)
            if delta > 0:
                time.sleep(delta)
        return True

    def set_joint_direct(self, joint=None):
        self.last_joint_cmd = joint
        if joint is None:
            joint = []
        if self.comp_hysteresis:  # update new joint values
            joint = self.hyst_comp.step(joint)
        return super().set_joint_direct(joint=joint)

    def set_joint(self, joint=None, wait_callback=True, use_interpolation=False):
        self.last_joint_cmd = joint
        if joint is None:
            joint = []
        if self.comp_hysteresis:    # update new joint values
            if use_interpolation:   # use interpolated input to the hyst_comp model
                joint = self.hyst_comp.cal_interpolate(joint, mode='calibrate')
            else:
                joint = self.hyst_comp.step(joint)
        return super().set_joint(joint=joint, wait_callback=wait_callback)
        # return self.set_joint_interpolate(joint1=joint1, joint2=joint2, method='cubic')

    # def set_joint_interpolate(self, joint=[], method='cubic'):
    #     raise NotImplementedError
    #     # Define q0 and qf for arm 1
    #     q10 = self.arm1.get_current_joint()
    #     if len(joint) == 0:
    #         q1f = q10
    #     else:
    #         q1f = joint
    #
    #     # Define q0 and qf for arm 2
    #     q20 = self.arm2.get_current_joint()
    #     if len(joint2) == 0:
    #         q2f = q20
    #     else:
    #         q2f = joint2
    #
    #     # Define trajectory
    #     v_max = np.array([0.5, 0.5, 0.2, 2.0, 2.0, 2.0])*3
    #     a_max = np.array([0.5, 0.5, 0.2, 2.0, 2.0, 2.0])*3
    #     if method == 'cubic':
    #         time1, traj1 = self.arm1.cubic(q10, q1f, v_max=v_max, a_max=a_max, t_step=0.01)
    #         time2, traj2 = self.arm2.cubic(q20, q2f, v_max=v_max, a_max=a_max, t_step=0.01)
    #     elif method == 'LSPB':
    #         time1, traj1 = self.arm1.LSPB(q10, q1f, v_max=v_max, a_max=a_max, t_step=0.01)
    #         time2, traj2 = self.arm2.LSPB(q20, q2f, v_max=v_max, a_max=a_max, t_step=0.01)
    #     else:
    #         raise IndexError
    #
    #     # Execute trajectory
    #     len1 = len(traj1)
    #     len2 = len(traj2)
    #     len_max = max(len1, len2)
    #     traj_add1 = np.repeat([traj1[-1]], len_max - len1, axis=0)
    #     traj_add2 = np.repeat([traj2[-1]], len_max - len2, axis=0)
    #     traj1 = np.concatenate((traj1, traj_add1), axis=0)
    #     traj2 = np.concatenate((traj2, traj_add2), axis=0)
    #     for i, (q1, q2) in enumerate(zip(traj1, traj2)):
    #         super().set_joint(joint1=q1, joint2=q2, wait_callback=False)
    #         if self.stop_collision:
    #             if self.detect_collision():
    #                 # when collide, temporary make the threshold higher to escape in next move.
    #                 self.curr_threshold = self.curr_threshold_high
    #                 return False
    #         time.sleep(0.01)
    #     super().set_joint(joint1=traj1[-1], joint2=traj2[-1], wait_callback=True)    # wait until goal reached
    #     self.curr_threshold = self.curr_threshold_low
    #     return True

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
