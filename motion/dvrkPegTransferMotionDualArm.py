from FLSpegtransfer.motion.dvrkController import dvrkController
from FLSpegtransfer.motion.dvrkArm import dvrkArm
import FLSpegtransfer.motion.dvrkVariables as dvrkVar
from FLSpegtransfer.motion.dvrkKinematics import dvrkKinematics
from FLSpegtransfer.traj_opt.CubicOptimizer_1wp import CubicOptimizer_1wp
from FLSpegtransfer.traj_opt.CubicOptimizer_2wp import CubicOptimizer_2wp
from FLSpegtransfer.traj_opt.PegMotionOptimizer import PegMotionOptimizerV2b
import FLSpegtransfer.utils.CmnUtil as U
import numpy as np
import threading, time


class dvrkPegTransferMotionDualArm:
    """
    Motion library for peg transfer
    """
    def __init__(self, use_controller, use_optimization, optimizer):
        # motion library
        self.use_optimization = use_optimization
        self.optimizer = optimizer
        if use_controller:
            self.arm1 = dvrkController(arm_name='/PSM1', comp_hysteresis=True, stop_collision=False)
            self.arm2 = dvrkController(arm_name='/PSM2', comp_hysteresis=True, stop_collision=False)
            # self.arm2 = dvrkArm('/PSM2')
        else:
            self.arm1 = dvrkArm('/PSM1')
            self.arm2 = dvrkArm('/PSM2')
        if self.optimizer == 'cubic':
            self.motion_opt_2wp = CubicOptimizer_2wp(max_vel=dvrkVar.v_max, max_acc=dvrkVar.a_max,
                                                     t_step=0.01, print_out=False, visualize=False)
            self.motion_opt_1wp = CubicOptimizer_1wp(max_vel=dvrkVar.v_max, max_acc=dvrkVar.a_max,
                                                     t_step=0.01, print_out=False, visualize=False)
        elif self.optimizer == 'qp':
            self.motion_opt = PegMotionOptimizerV2b(max_vel=dvrkVar.v_max, max_acc=dvrkVar.a_max, t_step=0.01)
        else:
            raise ValueError

        # Motion variables
        self.jaw_opening = np.deg2rad([60, 65])  # PSM1, PSM2
        self.jaw_closing = np.deg2rad([0, -10])

        self.pos_org1 = [0.060, 0.0, -0.095]
        self.rot_org1 = [0.0, 0.0, 0.0, 1.0]
        self.jaw_org1 = self.jaw_closing[0]
        self.pos_org2 = [-0.060, 0.0, -0.095]
        self.rot_org2 = [0.0, 0.0, 0.0, 1.0]
        self.jaw_org2 = self.jaw_closing[1]

        self.offset_grasp = [-0.005, +0.002]
        self.height_ready = -0.120
        self.height_drop = -0.135

    def move_origin(self):
        t1 = threading.Thread(target=self.arm1.set_pose_interpolate, args=(self.pos_org1, self.rot_org1))
        t2 = threading.Thread(target=self.arm2.set_pose_interpolate, args=(self.pos_org2, self.rot_org2))
        t3 = threading.Thread(target=self.arm1.set_jaw_interpolate, args=[[self.jaw_org1]])
        t4 = threading.Thread(target=self.arm2.set_jaw_interpolate, args=[[self.jaw_org2]])
        t1.daemon = True
        t2.daemon = True
        t3.daemon = True
        t4.daemon = True
        t1.start(); t2.start(); t3.start(); t4.start()
        t1.join(); t2.join(); t3.join(); t4.join()

    def move_trajectory(self, obj, q_pos):
        for joint in q_pos:
            start = time.perf_counter()
            obj.set_joint_direct(joint=joint)
            end = time.perf_counter()
            delta = 0.01 - (end-start)
            if delta > 0:
                time.sleep(delta)

    def go_pick_traj(self, obj, pos_pick, rot_pick, optimized, optimizer):
        if optimized:
            # trajectory to pick
            q0 = obj.get_current_joint()
            qw = dvrkKinematics.pose_to_joint(pos=[pos_pick[0], pos_pick[1], self.height_ready],
                                              rot=U.euler_to_quaternion(np.deg2rad([rot_pick, 0.0, 0.0])))
            qf = dvrkKinematics.pose_to_joint(pos=[pos_pick[0], pos_pick[1], pos_pick[2] + self.offset_grasp[1]],
                                              rot=U.euler_to_quaternion(np.deg2rad([rot_pick, 0.0, 0.0])))
            if optimizer == 'cubic':
                q_pos, _ = self.motion_opt_1wp.optimize(q0, qw, qf)
            elif optimizer == 'qp':
                x, H = self.motion_opt.optimize_handover_to_drop_motion(q0, qw[0], qf[0])
                if x is not None:
                    dim = 6
                    q_pos = np.array([x[t * dim:(t + 1) * dim] for t in range(H + 1)])
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

    def go_pick(self, pos_pick1, rot_pick1, pos_pick2, rot_pick2):
        q_pos1 = self.go_pick_traj(self.arm1, pos_pick1, rot_pick1, self.use_optimization, self.optimizer)
        q_pos2 = self.go_pick_traj(self.arm2, pos_pick2, rot_pick2, self.use_optimization, self.optimizer)

        # be ready to place & close jaw
        t2 = threading.Thread(target=self.move_trajectory, args=(self.arm2, q_pos2))
        t1 = threading.Thread(target=self.move_trajectory, args=(self.arm1, q_pos1))
        t3 = threading.Thread(target=self.arm1.set_jaw_interpolate, args=[[self.jaw_closing[0]]])
        t4 = threading.Thread(target=self.arm2.set_jaw_interpolate, args=[[self.jaw_closing[1]]])
        t1.daemon = True
        t2.daemon = True
        t3.daemon = True
        t4.daemon = True
        t1.start(); t2.start(); t3.start(); t4.start()
        t1.join(); t2.join(); t3.join(); t4.join()

    def transfer_block_traj(self, pos_pick, rot_pick, pos_place, rot_place, optimized, optimizer):
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
            if optimizer == 'cubic':
                # q_pos, _ = self.motion_opt_2wp.optimize(q0, qw1, qw2, qf, dqw1, dqw2)
                q_pos, _ = self.motion_opt_1wp.optimize(q0, qw1, qw2)
            elif optimizer == 'qp':
                x, H = self.motion_opt.optimize_lift_to_handover_motion(q0[0], qw1[0], qw2[0])
                if x is not None:
                    dim = 6
                    q_pos = np.array([x[t * dim:(t + 1) * dim] for t in range(H + 1)])
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

            # posf = [pos_place[0], pos_place[1], self.height_drop]
            # rotf = [np.deg2rad(rot_place), 0.0, 0.0]
            # posef = np.concatenate((posf, rotf))

            # Define trajectory
            _, q_pos1 = dvrkArm.cubic_cartesian(pose0, posew1, vel_limit=dvrkVar.v_max, acc_limit=dvrkVar.a_max,
                                                tf_init=0.5, t_step=0.01)
            _, q_pos2 = dvrkArm.cubic_cartesian(posew1, posew2, vel_limit=dvrkVar.v_max, acc_limit=dvrkVar.a_max,
                                                tf_init=0.5, t_step=0.01)
            # _, q_pos3 = dvrkArm.cubic_cartesian(posew2, posef, vel_limit=dvrkVar.v_max, acc_limit=dvrkVar.a_max,
            #                                     tf_init=0.5, t_step=0.01)
            q_pos = np.concatenate((q_pos1, q_pos2))
            # q_pos = np.concatenate((q_pos1, q_pos2, q_pos3))
        return q_pos

    def transfer_block(self, pos_pick1, rot_pick1, pos_place1, rot_place1, pos_pick2, rot_pick2, pos_place2, rot_place2):
        # open jaw
        t3 = threading.Thread(target=self.arm1.set_jaw_interpolate, args=[[self.jaw_opening[0]]])
        t4 = threading.Thread(target=self.arm2.set_jaw_interpolate, args=[[self.jaw_opening[1]]])
        t3.daemon = True
        t4.daemon = True
        t3.start(); t4.start()
        t3.join(); t4.join()

        # go down toward block & close jaw
        t5 = threading.Thread(target=self.arm1.set_pose_interpolate,
                              args=([pos_pick1[0], pos_pick1[1], pos_pick1[2]+self.offset_grasp[0]], U.euler_to_quaternion(np.deg2rad([rot_pick1, 0.0, 0.0]))))
        t6 = threading.Thread(target=self.arm2.set_pose_interpolate,
                              args=([pos_pick2[0], pos_pick2[1], pos_pick2[2]+self.offset_grasp[0]], U.euler_to_quaternion(np.deg2rad([rot_pick2, 0.0, 0.0]))))
        t5.daemon = True
        t6.daemon = True
        t5.start(); t6.start()
        t5.join(); t6.join()

        t7 = threading.Thread(target=self.arm1.set_jaw_interpolate, args=[[self.jaw_closing[0]]])
        t8 = threading.Thread(target=self.arm2.set_jaw_interpolate, args=[[self.jaw_closing[1]]])
        t7.daemon = True
        t8.daemon = True
        t7.start(); t8.start()
        t7.join(); t8.join()

        # transfer block from peg to peg
        q_pos1 = self.transfer_block_traj(pos_pick1, rot_pick1, pos_place1, rot_place1, self.use_optimization, self.optimizer)
        q_pos2 = self.transfer_block_traj(pos_pick2, rot_pick2, pos_place2, rot_place2, self.use_optimization, self.optimizer)
        t1 = threading.Thread(target=self.move_trajectory, args=(self.arm1, q_pos1))
        t2 = threading.Thread(target=self.move_trajectory, args=(self.arm2, q_pos2))
        t3 = threading.Thread(target=self.arm1.set_jaw_interpolate, args=[[self.jaw_closing[0]]])
        t4 = threading.Thread(target=self.arm2.set_jaw_interpolate, args=[[self.jaw_closing[1]]])
        t1.daemon = True
        t2.daemon = True
        t3.daemon = True
        t4.daemon = True
        t1.start(); t2.start(); t3.start(); t4.start()
        t1.join(); t2.join(); t3.join(); t4.join()

        # open jaw
        # t3 = threading.Thread(target=self.arm1.set_jaw_interpolate, args=[[self.jaw_opening[0]]])
        # t4 = threading.Thread(target=self.arm2.set_jaw_interpolate, args=[[self.jaw_opening[1]]])
        # t3.daemon = True
        # t4.daemon = True
        # t3.start(); t4.start()
        # t3.join(); t4.join()

    def return_to_peg_traj(self, obj, pos_pick, rot_pick, optimized, optimizer):
        if optimized:
            # trajectory of returning to another peg to pick-up
            p0, quat0 = obj.get_current_pose()
            q0 = obj.get_current_joint()
            qw1 = dvrkKinematics.pose_to_joint(pos=[p0[0], p0[1], self.height_ready - 0.01], rot=quat0)
            qw2 = dvrkKinematics.pose_to_joint(pos=[pos_pick[0], pos_pick[1], self.height_ready],
                                               rot=U.euler_to_quaternion(np.deg2rad([rot_pick, 0.0, 0.0])))
            qf = dvrkKinematics.pose_to_joint(pos=[pos_pick[0], pos_pick[1], pos_pick[2] + self.offset_grasp[1]],
                                              rot=U.euler_to_quaternion(np.deg2rad([rot_pick, 0.0, 0.0])))
            if optimizer == 'cubic':
                q_pos, _ = self.motion_opt_2wp.optimize(q0, qw1, qw2, qf)
                # q_pos, _ = self.motion_opt_1wp.optimize(q0, qw1, qw2, dqw1)
            elif optimizer == 'qp':
                x, H = self.motion_opt.optimize_motion(q0, qw1[0], qw2[0], qf[0])
                if x is not None:
                    dim = 6
                    q_pos = np.array([x[t * dim:(t + 1) * dim] for t in range(H + 1)])
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

    def return_to_peg(self, pos_pick1, rot_pick1, pos_pick2, rot_pick2):
        # return to peg
        q_pos1 = self.return_to_peg_traj(self.arm1, pos_pick1, rot_pick1, self.use_optimization, self.optimizer)
        q_pos2 = self.return_to_peg_traj(self.arm2, pos_pick2, rot_pick2, self.use_optimization, self.optimizer)
        t1 = threading.Thread(target=self.move_trajectory, args=(self.arm1, q_pos1))
        t2 = threading.Thread(target=self.move_trajectory, args=(self.arm2, q_pos2))
        t3 = threading.Thread(target=self.arm1.set_jaw_interpolate, args=[[self.jaw_closing[0]]])
        t4 = threading.Thread(target=self.arm2.set_jaw_interpolate, args=[[self.jaw_closing[1]]])
        t1.daemon = True
        t2.daemon = True
        t3.daemon = True
        t4.daemon = True
        t1.start(); t2.start(); t3.start(); t4.start()
        t1.join(); t2.join(); t3.join(); t4.join()

        # open jaw
        t3 = threading.Thread(target=self.arm1.set_jaw_interpolate, args=[[self.jaw_opening[0]]])
        t4 = threading.Thread(target=self.arm2.set_jaw_interpolate, args=[[self.jaw_opening[1]]])
        t3.daemon = True
        t4.daemon = True
        t3.start(); t4.start()
        t3.join(); t4.join()

    def servoing_block(self, pos_place1, rot_place1, pos_place2, rot_place2):
        # go down toward block & open jaw
        t1 = threading.Thread(target=self.arm1.set_pose_interpolate,
                              args=([pos_place1[0], pos_place1[1], self.height_ready],
                                    U.euler_to_quaternion(np.deg2rad([rot_place1, 0.0, 0.0]))))
        t2 = threading.Thread(target=self.arm2.set_pose_interpolate,
                              args=([pos_place2[0], pos_place2[1], self.height_ready],
                                    U.euler_to_quaternion(np.deg2rad([rot_place2, 0.0, 0.0]))))
        t1.daemon = True
        t2.daemon = True
        t1.start(); t2.start()
        t1.join(); t2.join()

        t3 = threading.Thread(target=self.arm1.set_pose_interpolate,
                              args=([pos_place1[0], pos_place1[1], self.height_drop],
                                    U.euler_to_quaternion(np.deg2rad([rot_place1, 0.0, 0.0]))))
        t4 = threading.Thread(target=self.arm2.set_pose_interpolate,
                              args=([pos_place2[0], pos_place2[1], self.height_drop],
                                    U.euler_to_quaternion(np.deg2rad([rot_place2, 0.0, 0.0]))))
        t3.daemon = True
        t4.daemon = True
        t3.start(); t4.start()
        t3.join(); t4.join()

        t5 = threading.Thread(target=self.arm1.set_jaw_interpolate, args=[[self.jaw_opening[0]]])
        t6 = threading.Thread(target=self.arm2.set_jaw_interpolate, args=[[self.jaw_opening[1]]])
        t5.daemon = True
        t6.daemon = True
        t5.start(); t6.start()
        t5.join(); t6.join()


if __name__ == "__main__":
    motion = dvrkPegTransferMotionDualArm()
    motion.move_origin()