from FLSpegtransferHO.motion.dvrkController import dvrkController
from FLSpegtransferHO.motion.dvrkArm import dvrkArm
from FLSpegtransferHO.traj_opt.CubicOptimizer_1wp import CubicOptimizer_1wp
from FLSpegtransferHO.traj_opt.CubicOptimizer_2wp import CubicOptimizer_2wp
import FLSpegtransferHO.motion.dvrkVariables as dvrkVar
from FLSpegtransferHO.motion.dvrkKinematics import dvrkKinematics
import FLSpegtransferHO.utils.CmnUtil as U
import numpy as np
import threading, time


class dvrkPegTransferMotionSingleArm:
    """
    Motion library for peg transfer
    """
    def __init__(self, use_controller, use_optimization, which_arm):
        # motion library
        self.use_optimization = use_optimization
        if use_controller:
            self.arm1 = dvrkController(arm_name='/'+which_arm, comp_hysteresis=True)
        else:
            self.arm1 = dvrkArm(arm_name='/'+which_arm)
        self.motion_opt_2wp = CubicOptimizer_2wp()
        self.motion_opt_1wp = CubicOptimizer_1wp()

        # Motion variables
        if which_arm=='PSM1':
            self.pos_org = [0.060, 0.0, -0.095]
            self.rot_org = [0.0, 0.0, 0.0, 1.0]
            self.jaw_org = [0.0]
        elif which_arm=='PSM2':
            self.pos_org = [-0.060, 0.0, -0.095]
            self.rot_org = [0.0, 0.0, 0.0, 1.0]
            self.jaw_org = [0.0]
        else:
            raise ValueError

        self.offset_grasp = [-0.005, +0.002]
        self.height_ready = -0.125
        self.height_drop = -0.140
        self.jaw_opening = [np.deg2rad(50)]
        self.jaw_closing = [np.deg2rad(-10)]

    def move_origin(self):
        t1 = threading.Thread(target=self.arm1.set_pose_interpolate, args=(self.pos_org, self.rot_org))
        t2 = threading.Thread(target=self.arm1.set_jaw_interpolate, args=[self.jaw_org])
        t1.daemon = True
        t2.daemon = True
        t1.start(); t2.start()
        t1.join(); t2.join()

    def move_trajectory(self, obj, q_pos):
        for joint in q_pos:
            start = time.perf_counter()
            obj.set_joint_direct(joint=joint)
            end = time.perf_counter()
            delta = 0.01 - (end - start)
            if delta > 0:
                time.sleep(delta)

    def go_pick_traj(self, obj, pos_pick, rot_pick, optimized):
        if optimized:
            # trajectory to pick
            q0 = obj.get_current_joint()
            qw = dvrkKinematics.pose_to_joint(pos=[pos_pick[0], pos_pick[1], self.height_ready],
                                               rot=U.euler_to_quaternion(np.deg2rad([rot_pick, 0.0, 0.0])))
            qf = dvrkKinematics.pose_to_joint(pos=[pos_pick[0], pos_pick[1], pos_pick[2] + self.offset_grasp[1]],
                                               rot=U.euler_to_quaternion(np.deg2rad([rot_pick, 0.0, 0.0])))
            J = dvrkKinematics.jacobian(qw)
            dvw = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
            dqw = np.linalg.inv(J).dot(dvw)
            q_pos, _ = self.motion_opt_1wp.optimize(q0, qw, qf, dqw,
                                                    max_vel=dvrkVar.v_max, max_acc=dvrkVar.a_max,
                                                    t_step=0.01, print_out=False, visualize=False)
        else:
            pos0, quat0 = obj.get_current_pose()
            rot0 = U.quaternion_to_euler(quat0)
            pose0 = np.concatenate((pos0, rot0))

            posw = [pos_pick[0], pos_pick[1], self.height_ready]
            rotw = [np.deg2rad(rot_pick), 0.0, 0.0]
            posew = np.concatenate((posw, rotw))

            posf = [pos_pick[0], pos_pick[1], pos_pick[2] + self.offset_grasp[1]]
            rotf = [np.deg2rad(rot_pick), 0.0, 0.0]
            posef = np.concatenate((posf, rotf))

            # Define trajectory
            _, q_pos1 = dvrkArm.cubic_cartesian(pose0, posew, vel_limit=dvrkVar.v_max, acc_limit=dvrkVar.a_max,
                                                tf_init=0.5, t_step=0.01)
            _, q_pos2 = dvrkArm.cubic_cartesian(posew, posef, vel_limit=dvrkVar.v_max, acc_limit=dvrkVar.a_max,
                                                tf_init=0.5, t_step=0.01)
            q_pos = np.concatenate((q_pos1, q_pos2))
        return q_pos

    def go_pick(self, pos_pick, rot_pick):
        q_pos = self.go_pick_traj(self.arm1, pos_pick, rot_pick, optimized=self.use_optimization)
        # be ready to place & close jaw

        t1 = threading.Thread(target=self.move_trajectory, args=(self.arm1, q_pos))
        t2 = threading.Thread(target=self.arm1.set_jaw_interpolate, args=[[self.jaw_closing[0]]])
        t1.daemon = True
        t2.daemon = True
        t1.start(); t2.start()
        t1.join(); t2.join()

    def transfer_block_traj(self, pos_pick, rot_pick, pos_place, rot_place, optimized):
        if optimized:
            # trajectory to transferring block from peg to peg
            q0 = dvrkKinematics.pose_to_joint(pos=[pos_pick[0], pos_pick[1], pos_pick[2] + self.offset_grasp[0]],
                                              rot=U.euler_to_quaternion(np.deg2rad([rot_pick, 0.0, 0.0])))
            qw1 = dvrkKinematics.pose_to_joint(pos=[pos_pick[0], pos_pick[1], self.height_ready],
                                               rot=U.euler_to_quaternion(np.deg2rad([rot_pick, 0.0, 0.0])))
            qw2 = dvrkKinematics.pose_to_joint(pos=[pos_place[0], pos_place[1], self.height_ready],
                                               rot=U.euler_to_quaternion(np.deg2rad([rot_place, 0.0, 0.0])))
            qf = dvrkKinematics.pose_to_joint(pos=[pos_place[0], pos_place[1], self.height_drop],
                                              rot=U.euler_to_quaternion(np.deg2rad([rot_place, 0.0, 0.0])))
            J1 = dvrkKinematics.jacobian(qw1)
            J2 = dvrkKinematics.jacobian(qw2)
            dvw = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
            dqw1 = np.linalg.inv(J1).dot(dvw)
            dqw2 = np.linalg.inv(J2).dot(dvw)
            q_pos, _ = self.motion_opt_2wp.optimize(q0, qw1, qw2, qf, dqw1, dqw2,
                                                    max_vel=dvrkVar.v_max, max_acc=dvrkVar.a_max,
                                                    t_step=0.01, print_out=False, visualize=False)
        else:
            # trajectory to transferring block from peg to peg
            pos0 = [pos_pick[0], pos_pick[1], pos_pick[2] + self.offset_grasp[0]]
            rot0 = [np.deg2rad(rot_pick), 0.0, 0.0]
            pose0 = np.concatenate((pos0, rot0))

            posw1 = [pos_pick[0], pos_pick[1], self.height_ready]
            rotw1 = [np.deg2rad(rot_pick), 0.0, 0.0]
            posew1 = np.concatenate((posw1, rotw1))

            posw2 = [pos_place[0], pos_place[1], self.height_ready]
            rotw2 = [np.deg2rad(rot_place), 0.0, 0.0]
            posew2 = np.concatenate((posw2, rotw2))

            posf = [pos_place[0], pos_place[1], self.height_drop]
            rotf = [np.deg2rad(rot_place), 0.0, 0.0]
            posef = np.concatenate((posf, rotf))

            # Define trajectory
            _, q_pos1 = dvrkArm.cubic_cartesian(pose0, posew1, vel_limit=dvrkVar.v_max, acc_limit=dvrkVar.a_max,
                                                tf_init=0.5, t_step=0.01)
            _, q_pos2 = dvrkArm.cubic_cartesian(posew1, posew2, vel_limit=dvrkVar.v_max, acc_limit=dvrkVar.a_max,
                                                tf_init=0.5, t_step=0.01)
            _, q_pos3 = dvrkArm.cubic_cartesian(posew2, posef, vel_limit=dvrkVar.v_max, acc_limit=dvrkVar.a_max,
                                                tf_init=0.5, t_step=0.01)
            q_pos = np.concatenate((q_pos1, q_pos2, q_pos3))
        return q_pos

    def transfer_block(self, pos_pick, rot_pick, pos_place, rot_place):
        # open jaw
        self.arm1.set_jaw_interpolate(self.jaw_opening)

        # go down toward block & close jaw
        self.arm1.set_pose_interpolate(pos=[pos_pick[0], pos_pick[1], pos_pick[2]+self.offset_grasp[0]],
                                       rot=U.euler_to_quaternion(np.deg2rad([rot_pick, 0.0, 0.0])))
        self.arm1.set_jaw_interpolate(self.jaw_closing)

        # transfer block from peg to peg
        q_pos = self.transfer_block_traj(pos_pick, rot_pick, pos_place, rot_place, self.use_optimization)
        self.move_trajectory(self.arm1, q_pos)

        # open jaw
        self.arm1.set_jaw_interpolate(self.jaw_opening)

    def return_to_peg_traj(self, obj, pos_pick, rot_pick, optimized):
        # trajectory of returning to another peg to pick-up
        if optimized:
            p0, quat0 = self.arm1.get_current_pose()
            q0 = self.arm1.get_current_joint()
            qw1 = dvrkKinematics.pose_to_joint(pos=[p0[0], p0[1], self.height_ready], rot=quat0)
            qw2 = dvrkKinematics.pose_to_joint(pos=[pos_pick[0], pos_pick[1], self.height_ready],
                                                rot=U.euler_to_quaternion(np.deg2rad([rot_pick, 0.0, 0.0])))
            qf = dvrkKinematics.pose_to_joint(pos=[pos_pick[0], pos_pick[1], pos_pick[2] + self.offset_grasp[1]],
                                               rot=U.euler_to_quaternion(np.deg2rad([rot_pick, 0.0, 0.0])))
            J1 = dvrkKinematics.jacobian(qw1)
            J2 = dvrkKinematics.jacobian(qw2)
            dvw = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
            dqw1 = np.linalg.inv(J1).dot(dvw)
            dqw2 = np.linalg.inv(J2).dot(dvw)
            q_pos, _ = self.motion_opt_2wp.optimize(q0, qw1, qw2, qf, dqw1, dqw2,
                                                    max_vel=dvrkVar.v_max, max_acc=dvrkVar.a_max,
                                                    t_step=0.01, print_out=False, visualize=False)
        else:
            pos0, quat0 = obj.get_current_pose()
            rot0 = U.quaternion_to_euler(quat0)
            pose0 = np.concatenate((pos0, rot0))

            posw1 = [pos0[0], pos0[1], self.height_ready]
            rotw1 = rot0
            posew1 = np.concatenate((posw1, rotw1))

            posw2 = [pos_pick[0], pos_pick[1], self.height_ready]
            rotw2 = [np.deg2rad(rot_pick), 0.0, 0.0]
            posew2 = np.concatenate((posw2, rotw2))

            posf = [pos_pick[0], pos_pick[1], pos_pick[2] + self.offset_grasp[1]]
            rotf = [np.deg2rad(rot_pick), 0.0, 0.0]
            posef = np.concatenate((posf, rotf))

            # Define trajectory
            _, q_pos1 = dvrkArm.cubic_cartesian(pose0, posew1, vel_limit=dvrkVar.v_max, acc_limit=dvrkVar.a_max,
                                              tf_init=0.5, t_step=0.01)
            _, q_pos2 = dvrkArm.cubic_cartesian(posew1, posew2, vel_limit=dvrkVar.v_max, acc_limit=dvrkVar.a_max,
                                              tf_init=0.5, t_step=0.01)
            _, q_pos3 = dvrkArm.cubic_cartesian(posew2, posef, vel_limit=dvrkVar.v_max, acc_limit=dvrkVar.a_max,
                                              tf_init=0.5, t_step=0.01)
            q_pos = np.concatenate((q_pos1, q_pos2, q_pos3))
        return q_pos

    def return_to_peg(self, pos_pick, rot_pick):
        # return to peg
        q_pos = self.return_to_peg_traj(self.arm1, pos_pick, rot_pick, self.use_optimization)
        t1 = threading.Thread(target=self.move_trajectory, args=(self.arm1, q_pos))
        t2 = threading.Thread(target=self.arm1.set_jaw_interpolate, args=[self.jaw_closing])
        t1.daemon = True
        t2.daemon = True
        t1.start(); t2.start()
        t1.join(); t2.join()

        # open jaw
        self.arm1.set_jaw_interpolate(self.jaw_opening)


if __name__ == "__main__":
    motion = dvrkPegTransferMotionSingleArm()
    motion.move_origin()