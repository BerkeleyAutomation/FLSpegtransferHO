from FLSpegtransfer.motion.dvrkController import dvrkController
from FLSpegtransfer.motion.dvrkArm import dvrkArm
import FLSpegtransfer.utils.CmnUtil as U
from FLSpegtransfer.path import *
import numpy as np
import threading, time


class dvrkPegTransferMotion:
    """
    Motion library for peg transfer
    """
    def __init__(self):
        # motion library
        # self.dvrk = dvrkController(comp_hysteresis=True, stop_collision=False)
        self.dvrk = dvrkArm('/PSM1')

        # Motion variables
        self.pos_org1 = [0.060, 0.0, -0.095]
        self.rot_org1 = [0.0, 0.0, 0.0, 1.0]
        self.jaw_org1 = [0.0]
        self.pos_org2 = [-0.060, 0.0, -0.095]
        self.rot_org2 = [0.0, 0.0, 0.0, 1.0]
        self.jaw_org2 = [0.0]

        self.height_grasp_offset_above = +0.002
        self.height_grasp_offset_below = -0.005
        self.height_ready = -0.120
        self.height_drop = -0.133
        self.jaw_opening = [np.deg2rad(40)]
        self.jaw_opening_drop = [np.deg2rad(40)]
        self.jaw_closing = [np.deg2rad(0)]

    def move_origin(self):
        t1 = threading.Thread(target=self.dvrk.set_pose, args=(self.pos_org1, self.rot_org1))
        t2 = threading.Thread(target=self.dvrk.set_jaw_interpolate, args=[[self.jaw_org1]])
        t1.daemon = True
        t2.daemon = True
        t1.start(); t2.start()
        t1.join(); t2.join()

    def move_jaw(self, jaw='close'):
        if jaw=='close':
            jaw=self.jaw_closing
        elif jaw=='open':
            jaw=self.jaw_opening
        else:
            raise ValueError
        self.dvrk.set_jaw_interpolate(jaw=[jaw])

    # this function is used only when NN model uses history
    def move_random(self):
        which_arm = 'PSM1'
        filename = root + 'experiment/0_trajectory_extraction/' + which_arm + '/training_insertion_sampled_2000.npy'
        joint_traj = np.load(filename)
        for joint in joint_traj[:40]:
            self.dvrk.set_joint_dummy(joint1=joint)

    def pick_above_block(self, pos, rot):
        # go above block to pickup
        self.dvrk.set_pose_interpolate(pos=[pos[0], pos[1], self.height_ready],
                           rot=U.euler_to_quaternion(np.deg2rad([rot, 0.0, 0.0])))
        self.dvrk.set_jaw_interpolate(jaw=self.jaw_closing)

    def pick_block(self, pos, rot):
        # pos = [x,y,z]
        # rot = grasping angle
        # approach block & open jaw
        self.dvrk.set_pose_interpolate(pos=[pos[0], pos[1], pos[2]+self.height_grasp_offset_above],
                           rot=U.euler_to_quaternion(np.deg2rad([rot, 0.0, 0.0])))
        self.dvrk.set_jaw_interpolate(jaw=self.jaw_opening)

        # go down toward block & close jaw
        self.dvrk.set_pose_interpolate(pos=[pos[0], pos[1], pos[2]+self.height_grasp_offset_below],
                           rot=U.euler_to_quaternion(np.deg2rad([rot, 0.0, 0.0])))
        self.dvrk.set_jaw_interpolate(jaw=self.jaw_closing)

        # go up with block
        self.dvrk.set_pose_interpolate(pos=[pos[0], pos[1], self.height_ready],
                           rot=U.euler_to_quaternion(np.deg2rad([rot, 0.0, 0.0])))
        self.dvrk.set_jaw_interpolate(jaw=self.jaw_closing)

    def place_above_block(self, pos, rot):
        # be ready to place & close jaw
        self.dvrk.set_pose_interpolate(pos=[pos[0], pos[1], self.height_ready], rot=U.euler_to_quaternion(np.deg2rad([rot, 0.0, 0.0])))
        self.dvrk.set_jaw_interpolate(jaw=self.jaw_closing)

    def drop_block(self, pos, rot):
        # be ready to place & close jaw
        self.dvrk.set_pose_interpolate(pos=[pos[0], pos[1], self.height_ready], rot=U.euler_to_quaternion(np.deg2rad([rot, 0.0, 0.0])))
        self.dvrk.set_jaw_interpolate(jaw=self.jaw_closing)

        # go down toward peg & open jaw
        self.dvrk.set_pose_interpolate(pos=[pos[0], pos[1], self.height_drop], rot=U.euler_to_quaternion(np.deg2rad([rot, 0.0, 0.0])))
        self.dvrk.set_jaw_interpolate(jaw=self.jaw_opening_drop)

        # go up
        self.dvrk.set_pose_interpolate(pos=[pos[0], pos[1], self.height_ready], rot=U.euler_to_quaternion(np.deg2rad([rot, 0.0, 0.0])))
        self.dvrk.set_jaw_interpolate(jaw=self.jaw_opening_drop)


if __name__ == "__main__":
    motion = dvrkPegTransferMotion()
    motion.move_origin()