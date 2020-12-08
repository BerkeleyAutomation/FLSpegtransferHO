from FLSpegtransfer.motion.dvrkController import dvrkController
from FLSpegtransfer.motion.dvrkArm import dvrkArm
import FLSpegtransfer.motion.dvrkVariables as dvrkVar
from FLSpegtransfer.motion.dvrkKinematics import dvrkKinematics
from FLSpegtransfer.traj_opt.CubicOptimizer_1wp import CubicOptimizer_1wp
from FLSpegtransfer.traj_opt.CubicOptimizer_2wp import CubicOptimizer_2wp
import FLSpegtransfer.utils.CmnUtil as U
from FLSpegtransfer.path import *
import numpy as np
import threading, time


class dvrkPegTransferMotionDualArmOptimized:
    """
    Motion library for peg transfer
    """
    def __init__(self):
        # motion library
        self.arm1 = dvrkController(arm_name='/PSM1', comp_hysteresis=True, stop_collision=False)
        self.arm2 = dvrkController(arm_name='/PSM2', comp_hysteresis=True, stop_collision=False)
        self.use_controller = True
        # self.arm1 = dvrkArm('/PSM1')
        # self.arm2 = dvrkArm('/PSM2')
        self.dvrk_model = dvrkKinematics()
        self.motion_opt_2wp = CubicOptimizer_2wp()
        self.motion_opt_1wp = CubicOptimizer_1wp()

        # Motion variables
        self.jaw_opening = np.deg2rad([40, 40])  # PSM1, PSM2
        self.jaw_closing = np.deg2rad([0, 0])

        self.pos_org1 = [0.060, 0.0, -0.095]
        self.rot_org1 = [0.0, 0.0, 0.0, 1.0]
        self.jaw_org1 = self.jaw_closing[0]
        self.pos_org2 = [-0.060, 0.0, -0.095]
        self.rot_org2 = [0.0, 0.0, 0.0, 1.0]
        self.jaw_org2 = self.jaw_closing[0]

        self.offset_grasp = [-0.002, +0.003]
        self.height_ready = -0.120
        self.height_drop = -0.14

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
    def move_random(self):
        which_arm = 'PSM1'
        filename = root + 'experiment/0_trajectory_extraction/' + which_arm + '/training_insertion_sampled_2000.npy'
        joint_traj = np.load(filename)
        for joint in joint_traj[:40]:
            self.dvrk.set_joint_dummy(joint1=joint)

    def go_pick_traj(self, obj, pos_pick, rot_pick):
        # trajectory to pick
        q0 = obj.get_current_joint(wait_callback=True)
        qw = self.dvrk_model.pose_to_joint(pos=[pos_pick[0], pos_pick[1], self.height_ready - 0.005],
                                           rot=U.euler_to_quaternion(np.deg2rad([rot_pick, 0.0, 0.0])))
        qf = self.dvrk_model.pose_to_joint(pos=[pos_pick[0], pos_pick[1], pos_pick[2] + self.offset_grasp[1]],
                                           rot=U.euler_to_quaternion(np.deg2rad([rot_pick, 0.0, 0.0])))
        J = self.dvrk_model.jacobian(qw)
        dvw = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        dqw = np.linalg.inv(J).dot(dvw)
        q_pos, t = self.motion_opt_1wp.optimize(q0, qw, qf, dqw,
                                                max_vel=dvrkVar.v_max, max_acc=dvrkVar.a_max,
                                                t_step=0.01, print_out=False, visualize=False)
        return q_pos, t

    def go_pick_action(self, obj, q_pos):
        for joint in q_pos[:-1]:
            obj.set_joint(joint=joint, wait_callback=False)
            if self.use_controller:
                time.sleep(0.005)
            else:
                time.sleep(0.01)
        obj.set_joint(joint=q_pos[-1], wait_callback=True)

    def go_pick(self, pos_pick1, rot_pick1, pos_pick2, rot_pick2):
        q_pos1, t1 = self.go_pick_traj(self.arm1, pos_pick1, rot_pick1)
        q_pos2, t2 = self.go_pick_traj(self.arm2, pos_pick2, rot_pick2)

        # be ready to place & close jaw
        t1 = threading.Thread(target=self.go_pick_action, args=(self.arm1, q_pos1))
        t2 = threading.Thread(target=self.go_pick_action, args=(self.arm2, q_pos2))
        t3 = threading.Thread(target=self.arm1.set_jaw, args=[[self.jaw_closing[0]]])
        t4 = threading.Thread(target=self.arm2.set_jaw, args=[[self.jaw_closing[1]]])
        t1.daemon = True
        t2.daemon = True
        t3.daemon = True
        t4.daemon = True
        t1.start(); t2.start(); t3.start(); t4.start()
        t1.join(); t2.join(); t3.join(); t4.join()

    def transfer_block_traj(self, pos_pick, rot_pick, pos_place, rot_place):
        # trajectory to transferring block from peg to peg
        q0 = self.dvrk_model.pose_to_joint(pos=[pos_pick[0], pos_pick[1], pos_pick[2] + self.offset_grasp[0]],
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
        q_pos, t = self.motion_opt_2wp.optimize(q0, qw1, qw2, qf, dqw1, dqw2,
                                                max_vel=dvrkVar.v_max, max_acc=dvrkVar.a_max,
                                                t_step=0.01, print_out=False, visualize=False)
        return q_pos, t

    def transfer_block_action(self, obj, q_pos):
        for joint in q_pos[:-1]:
            obj.set_joint(joint=joint, wait_callback=False)
            if self.use_controller:
                time.sleep(0.005)
            else:
                time.sleep(0.01)
        obj.set_joint(joint=q_pos[-1], wait_callback=True)

    def transfer_block(self, pos_pick1, rot_pick1, pos_place1, rot_place1, pos_pick2, rot_pick2, pos_place2, rot_place2):
        # open jaw
        t3 = threading.Thread(target=self.arm1.set_jaw, args=[[self.jaw_opening[0]]])
        t4 = threading.Thread(target=self.arm2.set_jaw, args=[[self.jaw_opening[1]]])
        t3.daemon = True
        t4.daemon = True
        t3.start(); t4.start()
        t3.join(); t4.join()

        # go down toward block & close jaw
        t5 = threading.Thread(target=self.arm1.set_pose,
                              args=([pos_pick1[0], pos_pick1[1], pos_pick1[2]+self.offset_grasp[0]], U.euler_to_quaternion(np.deg2rad([rot_pick1, 0.0, 0.0]))))
        t6 = threading.Thread(target=self.arm2.set_pose,
                              args=([pos_pick2[0], pos_pick2[1], pos_pick2[2]+self.offset_grasp[0]], U.euler_to_quaternion(np.deg2rad([rot_pick2, 0.0, 0.0]))))
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

        # transfer block from peg to peg
        q_pos1, t1 = self.transfer_block_traj(pos_pick1, rot_pick1, pos_place1, rot_place1)
        q_pos2, t2 = self.transfer_block_traj(pos_pick2, rot_pick2, pos_place2, rot_place2)
        t1 = threading.Thread(target=self.transfer_block_action, args=(self.arm1, q_pos1))
        t2 = threading.Thread(target=self.transfer_block_action, args=(self.arm2, q_pos2))
        t3 = threading.Thread(target=self.arm1.set_jaw, args=[[self.jaw_closing[0]]])
        t4 = threading.Thread(target=self.arm2.set_jaw, args=[[self.jaw_closing[1]]])
        t1.daemon = True
        t2.daemon = True
        t3.daemon = True
        t4.daemon = True
        t1.start(); t2.start(); t3.start(); t4.start()
        t1.join(); t2.join(); t3.join(); t4.join()

        # open jaw
        t3 = threading.Thread(target=self.arm1.set_jaw, args=[[self.jaw_opening[0]]])
        t4 = threading.Thread(target=self.arm2.set_jaw, args=[[self.jaw_opening[1]]])
        t3.daemon = True
        t4.daemon = True
        t3.start(); t4.start()
        t3.join(); t4.join()

    def return_to_peg_traj(self, obj, pos_pick, rot_pick):
        # trajectory of returning to another peg to pick-up
        p0, quat0 = obj.get_current_pose()
        q0 = obj.get_current_joint(wait_callback=True)
        qw1 = self.dvrk_model.pose_to_joint(pos=[p0[0], p0[1], self.height_ready - 0.01], rot=quat0)
        qw2 = self.dvrk_model.pose_to_joint(pos=[pos_pick[0], pos_pick[1], self.height_ready - 0.005],
                                            rot=U.euler_to_quaternion(np.deg2rad([rot_pick, 0.0, 0.0])))
        qf = self.dvrk_model.pose_to_joint(pos=[pos_pick[0], pos_pick[1], pos_pick[2] + self.offset_grasp[1]],
                                           rot=U.euler_to_quaternion(np.deg2rad([rot_pick, 0.0, 0.0])))
        J1 = self.dvrk_model.jacobian(qw1)
        J2 = self.dvrk_model.jacobian(qw2)
        dvw = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        dqw1 = np.linalg.inv(J1).dot(dvw)
        dqw2 = np.linalg.inv(J2).dot(dvw)
        q_pos, t = self.motion_opt_2wp.optimize(q0, qw1, qw2, qf, dqw1, dqw2,
                                                max_vel=dvrkVar.v_max, max_acc=dvrkVar.a_max,
                                                t_step=0.01, print_out=False, visualize=False)
        return q_pos, t

    def return_to_peg_action(self, obj, q_pos):
        for joint in q_pos[:-1]:
            obj.set_joint(joint=joint, wait_callback=False)
            if self.use_controller:
                time.sleep(0.005)
            else:
                time.sleep(0.01)
        obj.set_joint(joint=q_pos[-1], wait_callback=True)

    def return_to_peg(self, pos_pick1, rot_pick1, pos_pick2, rot_pick2):
        # return to peg
        q_pos1, t1 = self.return_to_peg_traj(self.arm1, pos_pick1, rot_pick1)
        q_pos2, t2 = self.return_to_peg_traj(self.arm2, pos_pick2, rot_pick2)
        t1 = threading.Thread(target=self.return_to_peg_action, args=(self.arm1, q_pos1))
        t2 = threading.Thread(target=self.return_to_peg_action, args=(self.arm2, q_pos2))
        t3 = threading.Thread(target=self.arm1.set_jaw, args=[[self.jaw_closing[0]]])
        t4 = threading.Thread(target=self.arm2.set_jaw, args=[[self.jaw_closing[1]]])
        t1.daemon = True
        t2.daemon = True
        t3.daemon = True
        t4.daemon = True
        t1.start(); t2.start(); t3.start(); t4.start()
        t1.join(); t2.join(); t3.join(); t4.join()

        # open jaw
        t3 = threading.Thread(target=self.arm1.set_jaw, args=[[self.jaw_opening[0]]])
        t4 = threading.Thread(target=self.arm2.set_jaw, args=[[self.jaw_opening[1]]])
        t3.daemon = True
        t4.daemon = True
        t3.start(); t4.start()
        t3.join(); t4.join()


if __name__ == "__main__":
    motion = dvrkPegTransferMotionDualArmOptimized()
    motion.move_origin()