from FLSpegtransfer.motion.dvrkController import dvrkController
from FLSpegtransfer.motion.dvrkArm import dvrkArm
import FLSpegtransfer.utils.CmnUtil as U
from FLSpegtransfer.path import *
import numpy as np
import threading, time


class dvrkPegTransferMotionDualArm:
    """
    Motion library for peg transfer
    """
    def __init__(self):
        # motion library
        # self.dvrk = dvrkController(comp_hysteresis=True, stop_collision=False)
        self.arm1 = dvrkArm('/PSM1')
        self.arm2 = dvrkArm('/PSM2')

        # Motion variables
        self.jaw_opening = np.deg2rad([40, 40])  # PSM1, PSM2
        self.jaw_closing = np.deg2rad([0, 0])

        self.pos_org1 = [0.060, 0.0, -0.095]
        self.rot_org1 = [0.0, 0.0, 0.0, 1.0]
        self.jaw_org1 = self.jaw_closing[0]
        self.pos_org2 = [-0.060, 0.0, -0.095]
        self.rot_org2 = [0.0, 0.0, 0.0, 1.0]
        self.jaw_org2 = self.jaw_closing[1]

        self.offset_grasp = [-0.005, +0.002]
        self.height_ready = -0.120
        self.height_drop = -0.133

    # this function is used only when NN model uses history
    def move_random(self):
        which_arm = 'PSM1'
        filename = root + 'experiment/0_trajectory_extraction/' + which_arm + '/training_insertion_sampled_2000.npy'
        joint_traj = np.load(filename)
        for joint in joint_traj[:40]:
            self.dvrk.set_joint_dummy(joint1=joint)

    def move_origin(self):
        t1 = threading.Thread(target=self.arm1.set_pose_interpolate, args=(self.pos_org1, self.rot_org1))
        t2 = threading.Thread(target=self.arm2.set_pose_interpolate, args=(self.pos_org2, self.rot_org2))
        t3 = threading.Thread(target=self.arm1.set_jaw, args=[[self.jaw_org1]])
        t4 = threading.Thread(target=self.arm2.set_jaw, args=[[self.jaw_org2]])
        t1.daemon = True
        t2.daemon = True
        t3.daemon = True
        t4.daemon = True
        t1.start(); t2.start(); t3.start(); t4.start()
        t1.join(); t2.join(); t3.join(); t4.join()

    def move_jaw(self, jaw='close', which_arm='PSM1'):
        if which_arm == 'PSM1' or which_arm == 0:
            obj = self.arm1
            index = 0
        elif which_arm == 'PSM2' or which_arm == 1:
            obj = self.arm2
            index = 1
        else:
            raise ValueError
        if jaw == 'close':
            jaw = self.jaw_closing[index]
        elif jaw == 'open':
            jaw = self.jaw_opening[index]
        else:
            raise ValueError
        obj.set_jaw(jaw=[jaw])

    def move_above_block(self, pos1, rot1, pos2, rot2):
        # be ready to place & close jaw
        t1 = threading.Thread(target=self.arm1.set_pose_interpolate,
                              args=([pos1[0], pos1[1], self.height_ready], U.euler_to_quaternion(np.deg2rad([rot1, 0.0, 0.0]))))
        t2 = threading.Thread(target=self.arm2.set_pose_interpolate,
                              args=([pos2[0], pos2[1], self.height_ready], U.euler_to_quaternion(np.deg2rad([rot2, 0.0, 0.0]))))
        t3 = threading.Thread(target=self.arm1.set_jaw, args=[[self.jaw_closing[0]]])
        t4 = threading.Thread(target=self.arm2.set_jaw, args=[[self.jaw_closing[1]]])
        t1.daemon = True
        t2.daemon = True
        t3.daemon = True
        t4.daemon = True
        t1.start(); t2.start(); t3.start(); t4.start()
        t1.join(); t2.join(); t3.join(); t4.join()

    def pick_block(self, pos1, rot1, pos2, rot2):
        # pos = [x,y,z], rot = grasping angle
        # approach block
        t1 = threading.Thread(target=self.arm1.set_pose_interpolate,
                              args=([pos1[0], pos1[1], pos1[2]+self.offset_grasp[1]], U.euler_to_quaternion(np.deg2rad([rot1, 0.0, 0.0]))))
        t2 = threading.Thread(target=self.arm2.set_pose_interpolate,
                              args=([pos2[0], pos2[1], pos2[2]+self.offset_grasp[1]], U.euler_to_quaternion(np.deg2rad([rot2, 0.0, 0.0]))))
        t1.daemon = True
        t2.daemon = True
        t1.start(); t2.start()
        t1.join(); t2.join()

        # open jaw
        t3 = threading.Thread(target=self.arm1.set_jaw, args=[[self.jaw_opening[0]]])
        t4 = threading.Thread(target=self.arm2.set_jaw, args=[[self.jaw_opening[1]]])
        t3.daemon = True
        t4.daemon = True
        t3.start(); t4.start()
        t3.join(); t4.join()

        # go down toward block & close jaw
        t5 = threading.Thread(target=self.arm1.set_pose_interpolate,
                              args=([pos1[0], pos1[1], pos1[2]+self.offset_grasp[0]], U.euler_to_quaternion(np.deg2rad([rot1, 0.0, 0.0]))))
        t6 = threading.Thread(target=self.arm2.set_pose_interpolate,
                              args=([pos2[0], pos2[1], pos2[2]+self.offset_grasp[0]], U.euler_to_quaternion(np.deg2rad([rot2, 0.0, 0.0]))))
        t5.daemon = True
        t6.daemon = True
        t5.start(); t6.start()
        t5.join(); t6.join()

        t7 = threading.Thread(target=self.arm1.set_jaw, args=[[self.jaw_closing[0]]])
        t8 = threading.Thread(target=self.arm2.set_jaw, args=[[self.jaw_closing[1]]])
        t7.daemon = True
        t8.daemon = True
        t7.start(); t8.start()
        t7.join(); t8.join()

        # go up with block
        t9 = threading.Thread(target=self.arm1.set_pose_interpolate,
                              args=([pos1[0], pos1[1], self.height_ready], U.euler_to_quaternion(np.deg2rad([rot1, 0.0, 0.0]))))
        t10 = threading.Thread(target=self.arm2.set_pose_interpolate,
                              args=([pos2[0], pos2[1], self.height_ready], U.euler_to_quaternion(np.deg2rad([rot2, 0.0, 0.0]))))
        t9.daemon = True
        t10.daemon = True
        t9.start(); t10.start()
        t9.join(); t10.join()

    def drop_block(self, pos1, rot1, pos2, rot2):
        # be ready to place with jaw closing
        t1 = threading.Thread(target=self.arm1.set_pose_interpolate,
                              args=([pos1[0], pos1[1], self.height_ready], U.euler_to_quaternion(np.deg2rad([rot1, 0.0, 0.0]))))
        t2 = threading.Thread(target=self.arm2.set_pose_interpolate,
                              args=([pos2[0], pos2[1], self.height_ready], U.euler_to_quaternion(np.deg2rad([rot2, 0.0, 0.0]))))
        t1.daemon = True
        t2.daemon = True
        t1.start(); t2.start()
        t1.join(); t2.join()

        # go down toward peg
        t3 = threading.Thread(target=self.arm1.set_pose_interpolate,
                              args=([pos1[0], pos1[1], self.height_drop], U.euler_to_quaternion(np.deg2rad([rot1, 0.0, 0.0]))))
        t4 = threading.Thread(target=self.arm2.set_pose_interpolate,
                              args=([pos2[0], pos2[1], self.height_drop], U.euler_to_quaternion(np.deg2rad([rot2, 0.0, 0.0]))))
        t3.daemon = True
        t4.daemon = True
        t3.start(); t4.start()
        t3.join(); t4.join()

        # # open jaw
        t5 = threading.Thread(target=self.arm1.set_jaw, args=[[self.jaw_opening[0]]])
        t6 = threading.Thread(target=self.arm2.set_jaw, args=[[self.jaw_opening[1]]])
        t5.daemon = True
        t6.daemon = True
        t5.start(); t6.start()
        t5.join(); t6.join()

        # go up
        t7 = threading.Thread(target=self.arm1.set_pose_interpolate,
                              args=([pos1[0], pos1[1], self.height_ready], U.euler_to_quaternion(np.deg2rad([rot1, 0.0, 0.0]))))
        t8 = threading.Thread(target=self.arm2.set_pose_interpolate,
                              args=([pos2[0], pos2[1], self.height_ready], U.euler_to_quaternion(np.deg2rad([rot2, 0.0, 0.0]))))
        t7.daemon = True
        t8.daemon = True
        t7.start(); t8.start()
        t7.join(); t8.join()


if __name__ == "__main__":
    motion = dvrkPegTransferMotionDualArm()
    motion.move_origin()