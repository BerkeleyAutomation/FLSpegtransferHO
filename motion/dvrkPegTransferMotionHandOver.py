from FLSpegtransfer.motion.dvrkController import dvrkController
from FLSpegtransfer.motion.dvrkDualArm import dvrkDualArm
from FLSpegtransfer.motion.dvrkArm import dvrkArm
import FLSpegtransfer.utils.CmnUtil as U
from FLSpegtransfer.path import *
import numpy as np
import threading


class dvrkPegTransferMotionHandOver():
    """
    Motion library for peg transfer
    """
    def __init__(self):
        # motion library
        # self.dvrk = dvrkController(comp_hysteresis=True, stop_collision=False)
        self.arm1 = dvrkArm('/PSM1')
        self.arm2 = dvrkArm('/PSM2')

        # Motion variables
        self.jaw_opening = np.deg2rad([40, 40])     # PSM1, PSM2
        self.jaw_closing = np.deg2rad([0, 0])

        self.pos_org1 = [0.060, 0.0, -0.095]
        self.rot_org1 = [0.0, 0.0, 0.0, 1.0]
        self.jaw_org1 = self.jaw_closing[0]
        self.pos_org2 = [-0.060, 0.0, -0.095]
        self.rot_org2 = [0.0, 0.0, 0.0, 1.0]
        self.jaw_org2 = self.jaw_closing[1]

        self.height_ready = -0.120
        self.height_ready_handover = self.height_ready + 0.02
        self.height_drop = self.height_ready - 0.015
        self.offset_grasp = [-0.003, +0.003]
        self.offset_handover = +0.015   # the amount of move to hand off

    def move_origin(self):
        t1 = threading.Thread(target=self.arm1.set_pose, args=(self.pos_org1, self.rot_org1))
        t2 = threading.Thread(target=self.arm2.set_pose, args=(self.pos_org2, self.rot_org2))
        t3 = threading.Thread(target=self.arm1.set_jaw, args=[[self.jaw_org1]])
        t4 = threading.Thread(target=self.arm2.set_jaw, args=[[self.jaw_org2]])
        t1.daemon = True
        t2.daemon = True
        t3.daemon = True
        t4.daemon = True
        t1.start(); t2.start(); t3.start(); t4.start()
        t1.join(); t2.join(); t3.join(); t4.join()

    # this function is used only when NN model uses history
    def move_random(self, which_arm='PSM1'):
        filename = root + 'experiment/0_trajectory_extraction/' + which_arm + '/training_insertion_sampled_2000.npy'
        joint_traj = np.load(filename)
        for joint in joint_traj[:40]:
            self.dvrk.set_joint_dummy(joint1=joint)

    def move_jaw(self, jaw='close', which_arm='PSM1'):
        if which_arm=='PSM1' or which_arm==0:
            obj = self.arm1
            index = 0
        elif which_arm=='PSM2' or which_arm==1:
            obj = self.arm2
            index = 1
        else:
            raise ValueError
        if jaw=='close':
            jaw=self.jaw_closing[index]
        elif jaw=='open':
            jaw=self.jaw_opening[index]
        else:
            raise ValueError
        obj.set_jaw(jaw=[jaw])

    def move_upright(self, pos, rot, jaw='close', which_arm='PSM1', interpolate=False):
        if which_arm=='PSM1' or which_arm==0:
            obj = self.arm1
            index = 0
        elif which_arm=='PSM2' or which_arm==1:
            obj = self.arm2
            index = 1
        else:
            raise ValueError
        pos = [pos[0], pos[1], pos[2]]
        if rot==[]: pass
        else:   rot = U.euler_to_quaternion(np.deg2rad([rot, 0.0, 0.0]))
        if jaw=='close':
            jaw=self.jaw_closing[index]
        elif jaw=='open':
            jaw=self.jaw_opening[index]
        else:
            raise ValueError

        if interpolate:
            t1 = threading.Thread(target=obj.set_pose_interpolate, args=(pos, rot))
        else:
            t1 = threading.Thread(target=obj.set_pose, args=(pos, rot))
        t2 = threading.Thread(target=obj.set_jaw, args=[[jaw]])
        t1.daemon = True
        t2.daemon = True
        t1.start(); t2.start()
        t1.join(); t2.join()

    def pick_block(self, pos, rot, which_arm='PSM1'):
        if which_arm=='PSM1' or which_arm==0:
            obj = self.arm1
            index = 0
        elif which_arm=='PSM2' or which_arm==1:
            obj = self.arm2
            index = 1
        else:
            raise ValueError
        # approach block & open jaw
        obj.set_pose_interpolate(pos=[pos[0], pos[1], pos[2]+self.offset_grasp[1]],
                     rot=U.euler_to_quaternion(np.deg2rad([rot, 0.0, 0.0])))
        obj.set_jaw(jaw=[self.jaw_opening[index]])

        # go down toward block & close jaw
        obj.set_pose_interpolate(pos=[pos[0], pos[1], pos[2]+self.offset_grasp[0]],
                     rot=U.euler_to_quaternion(np.deg2rad([rot, 0.0, 0.0])))
        obj.set_jaw(jaw=[self.jaw_closing[index]])

    def drop_block(self, pos, rot, which_arm='PSM2'):
        if which_arm=='PSM1' or which_arm==0:
            obj = self.arm1
            index = 0
        elif which_arm=='PSM2' or which_arm==1:
            obj = self.arm2
            index = 1
        else:
            raise ValueError
        # be ready to place with jaw closing
        obj.set_pose(pos=[pos[0], pos[1], self.height_ready], rot=U.euler_to_quaternion(np.deg2rad([rot, 0.0, 0.0])))
        obj.set_jaw(jaw=[self.jaw_closing[index]])

        # go down toward peg & open jaw
        obj.set_pose(pos=[pos[0], pos[1], self.height_drop], rot=U.euler_to_quaternion(np.deg2rad([rot, 0.0, 0.0])))
        obj.set_jaw(jaw=[self.jaw_opening[index]])

        # go up
        obj.set_pose(pos=[pos[0], pos[1], self.height_ready], rot=U.euler_to_quaternion(np.deg2rad([rot, 0.0, 0.0])))
        obj.set_jaw(jaw=[self.jaw_opening[index]])

if __name__ == "__main__":
    motion = dvrkPegTransferMotionHandOver()
    motion.move_origin()