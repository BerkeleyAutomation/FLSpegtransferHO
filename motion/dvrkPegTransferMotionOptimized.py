from FLSpegtransfer.motion.dvrkController import dvrkController
from FLSpegtransfer.motion.dvrkArm import dvrkArm
from FLSpegtransfer.traj_opt.PegMotionOptimizer_2wp import PegMotionOptimizer_2wp
from FLSpegtransfer.traj_opt.PegMotionOptimizer_1wp import PegMotionOptimizer_1wp
import FLSpegtransfer.motion.dvrkVariables as dvrkVar
from FLSpegtransfer.motion.dvrkKinematics import dvrkKinematics
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
        self.dvrk_model = dvrkKinematics()
        self.motion_opt_2wp = PegMotionOptimizer_2wp()
        self.motion_opt_1wp = PegMotionOptimizer_1wp()

        # Motion variables
        self.pos_org1 = [0.060, 0.0, -0.095]
        self.rot_org1 = [0.0, 0.0, 0.0, 1.0]
        self.jaw_org1 = [0.0]
        self.pos_org2 = [-0.060, 0.0, -0.095]
        self.rot_org2 = [0.0, 0.0, 0.0, 1.0]
        self.jaw_org2 = [0.0]

        self.offset_grasp = [-0.005, +0.002]
        self.height_ready = -0.120
        self.height_drop = -0.14
        self.jaw_opening = [np.deg2rad(40)]
        self.jaw_closing = [np.deg2rad(0)]

    def move_origin(self):
        t1 = threading.Thread(target=self.dvrk.set_pose, args=(self.pos_org1, self.rot_org1))
        t2 = threading.Thread(target=self.dvrk.set_jaw_interpolate, args=[[self.jaw_org1]])
        t1.daemon = True
        t2.daemon = True
        t1.start(); t2.start()
        t1.join(); t2.join()

    # this function is used only when NN model uses history
    def move_random(self):
        which_arm = 'PSM1'
        filename = root + 'experiment/0_trajectory_extraction/' + which_arm + '/training_insertion_sampled_2000.npy'
        joint_traj = np.load(filename)
        for joint in joint_traj[:40]:
            self.dvrk.set_joint_dummy(joint1=joint)

    def move_jaw(self, jaw='close'):
        if jaw=='close':
            jaw=self.jaw_closing
        elif jaw=='open':
            jaw=self.jaw_opening
        else:
            raise ValueError
        self.dvrk.set_jaw_interpolate(jaw=[jaw])

    def go_pick(self, pos_pick, rot_pick):
        self.dvrk.set_jaw(jaw=self.jaw_closing)

        # trajectory to pick
        q0 = self.dvrk.get_current_joint(wait_callback=True)
        qw = self.dvrk_model.pose_to_joint(pos=[pos_pick[0], pos_pick[1], self.height_ready - 0.005],
                                           rot=U.euler_to_quaternion(np.deg2rad([rot_pick, 0.0, 0.0])))
        qf = self.dvrk_model.pose_to_joint(pos=[pos_pick[0], pos_pick[1], pos_pick[2] + self.offset_grasp[1]],
                                           rot=U.euler_to_quaternion(np.deg2rad([rot_pick, 0.0, 0.0])))
        J = self.dvrk_model.jacobian(qw)
        dvw = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        dqw = np.linalg.inv(J).dot(dvw)
        q_pos, q_vel, q_acc, t =\
            self.motion_opt_1wp.optimize_motion(q0, qw, dqw, qf, max_vel=dvrkVar.v_max, max_acc=dvrkVar.a_max,
                                                t_step=0.01, horizon=30, print_out=True, visualize=False)
        for joint in q_pos:
            self.dvrk.set_joint(joint=joint, wait_callback=False)
            time.sleep(0.01)
        self.dvrk.set_joint(joint=q_pos[-1], wait_callback=True)
        return q_pos, t

    def transfer_block(self, pos_pick, rot_pick, pos_place, rot_place):
        if rot_place == 0.0:    rot_place = 0.001   # otherwise, the optimizer gives an error of rank
        if rot_pick == 0.0:     rot_pick = 0.001
        # open jaw, go down toward block, and close jaw
        self.dvrk.set_jaw(jaw=self.jaw_opening)
        self.dvrk.set_pose(pos=[pos_pick[0], pos_pick[1], pos_pick[2]+self.offset_grasp[0]],
                           rot=U.euler_to_quaternion(np.deg2rad([rot_pick, 0.0, 0.0])))
        self.dvrk.set_jaw(jaw=self.jaw_closing)

        # trajectory to transferring block from peg to peg
        q0 = self.dvrk_model.pose_to_joint(pos=[pos_pick[0], pos_pick[1], pos_pick[2]+self.offset_grasp[0]],
                                           rot=U.euler_to_quaternion(np.deg2rad([rot_pick, 0.0, 0.0])))
        qw1 = self.dvrk_model.pose_to_joint(pos=[pos_pick[0], pos_pick[1], self.height_ready],
                                           rot=U.euler_to_quaternion(np.deg2rad([rot_pick, 0.0, 0.0])))
        qw2 = self.dvrk_model.pose_to_joint(pos=[pos_place[0], pos_place[1], self.height_ready],
                                           rot=U.euler_to_quaternion(np.deg2rad([rot_place, 0.0, 0.0])))
        qf = self.dvrk_model.pose_to_joint(pos=[pos_place[0], pos_place[1], self.height_drop],
                                           rot=U.euler_to_quaternion(np.deg2rad([rot_place, 0.0, 0.0])))
        J1 = self.dvrk_model.jacobian(qw1)
        J2 = self.dvrk_model.jacobian(qw2)
        dvw = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        dqw1 = np.linalg.inv(J1).dot(dvw)
        dqw2 = np.linalg.inv(J2).dot(dvw)
        print (q0)
        print(qw1)
        print(qw2)
        print(qf)
        print (dqw1)
        print(dqw2)
        q_pos, q_vel, q_acc, t =\
            self.motion_opt_2wp.optimize_motion(q0, qw1, dqw1, qw2, dqw2, qf,
                                                max_vel=dvrkVar.v_max, max_acc=dvrkVar.a_max, t_step=0.01,
                                                horizon=50, print_out=True, visualize=False)
        for joint in q_pos:
            self.dvrk.set_joint(joint=joint, wait_callback=False)
            time.sleep(0.01)
        self.dvrk.set_joint(joint=q_pos[-1], wait_callback=True)
        self.dvrk.set_jaw(jaw=self.jaw_opening)
        return q_pos, t

    def return_to_peg(self, pos_pick, rot_pick):
        p0, quat0 = self.dvrk.get_current_pose()
        self.dvrk.set_jaw(jaw=self.jaw_closing)

        # trajectory of returning to another peg to pick-up
        q0 = self.dvrk.get_current_joint(wait_callback=True)
        qw1 = self.dvrk_model.pose_to_joint(pos=[p0[0], p0[1], self.height_ready-0.01], rot=quat0)
        qw2 = self.dvrk_model.pose_to_joint(pos=[pos_pick[0], pos_pick[1], self.height_ready-0.005],
                                           rot=U.euler_to_quaternion(np.deg2rad([rot_pick, 0.0, 0.0])))
        qf = self.dvrk_model.pose_to_joint(pos=[pos_pick[0], pos_pick[1], pos_pick[2] + self.offset_grasp[1]],
                                           rot=U.euler_to_quaternion(np.deg2rad([rot_pick, 0.0, 0.0])))
        J1 = self.dvrk_model.jacobian(qw1)
        J2 = self.dvrk_model.jacobian(qw2)
        dvw = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        dqw1 = np.linalg.inv(J1).dot(dvw)
        dqw2 = np.linalg.inv(J2).dot(dvw)
        q_pos, q_vel, q_acc, t =\
            self.motion_opt_2wp.optimize_motion(q0, qw1, dqw1, qw2, dqw2, qf,
                                                max_vel=dvrkVar.v_max, max_acc=dvrkVar.a_max, t_step=0.01,
                                                horizon=50, print_out=True, visualize=False)
        for joint in q_pos:
            self.dvrk.set_joint(joint=joint, wait_callback=False)
            time.sleep(0.01)
        self.dvrk.set_joint(joint=q_pos[-1], wait_callback=True)
        self.dvrk.set_jaw(jaw=self.jaw_opening)
        return q_pos, t

    def return_to_origin(self, pos_place, rot_place):
        # trajectory to place
        self.dvrk.set_jaw(jaw=self.jaw_opening)
        q0 = self.dvrk_model.pose_to_joint(pos=[pos_place[0], pos_place[1], pos_place[2]+self.offset_grasp[0]],
                                           rot=U.euler_to_quaternion(np.deg2rad([rot_place, 0.0, 0.0])))
        qw = self.dvrk_model.pose_to_joint(pos=[pos_place[0], pos_place[1], self.height_ready],
                                           rot=U.euler_to_quaternion(np.deg2rad([rot_place, 0.0, 0.0])))
        qf = self.dvrk_model.pose_to_joint(pos=self.pos_org1, rot=self.rot_org1)
        J = self.dvrk_model.jacobian(qw)
        dvw = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        dqw = np.linalg.inv(J).dot(dvw)
        q_pos, q_vel, q_acc, t =\
            self.motion_opt_1wp.optimize_motion(q0, qw, dqw, qf, max_vel=dvrkVar.v_max, max_acc=dvrkVar.a_max,
                                                t_step=0.01, horizon=50, print_out=True, visualize=False)
        for joint in q_pos:
            self.dvrk.set_joint(joint=joint, wait_callback=False)
            time.sleep(0.01)
        self.dvrk.set_joint(joint=q_pos[-1], wait_callback=True)
        self.dvrk.set_jaw(jaw=self.jaw_closing)

    def drop_block(self, pos, rot):
        # be ready to place & close jaw
        self.dvrk.set_pose(pos=[pos[0], pos[1], self.height_ready], rot=U.euler_to_quaternion(np.deg2rad([rot, 0.0, 0.0])))
        self.dvrk.set_jaw(jaw=self.jaw_closing)

        # go down toward peg & open jaw
        self.dvrk.set_pose(pos=[pos[0], pos[1], self.height_drop],
                           rot=U.euler_to_quaternion(np.deg2rad([rot, 0.0, 0.0])))
        self.dvrk.set_jaw(jaw=self.jaw_opening)
        # pos_temp = [pos[0], pos[1], self.height_drop]
        # rot_temp = U.euler_to_quaternion(np.deg2rad([rot, 0.0, 0.0]))
        # t1 = threading.Thread(target=self.dvrk.set_pose, args=(pos_temp, rot_temp))
        # t2 = threading.Thread(target=self.dvrk.set_jaw_interpolate, args=[[self.jaw_opening]])
        # t1.daemon = True
        # t2.daemon = True
        # t1.start(); t2.start()
        # t1.join(); t2.join()

        # go up
        pos_temp = [pos[0], pos[1], self.height_ready]
        rot_temp = U.euler_to_quaternion(np.deg2rad([rot, 0.0, 0.0]))
        t1 = threading.Thread(target=self.dvrk.set_pose, args=(pos_temp, rot_temp))
        t2 = threading.Thread(target=self.dvrk.set_jaw_interpolate, args=[[self.jaw_closing]])
        t1.daemon = True
        t2.daemon = True
        t1.start(); t2.start()
        t1.join(); t2.join()


if __name__ == "__main__":
    motion = dvrkPegTransferMotion()
    motion.move_origin()