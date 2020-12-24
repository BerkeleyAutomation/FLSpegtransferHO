from FLSpegtransferHO.motion.dvrkController import dvrkController
from FLSpegtransferHO.motion.dvrkArm import dvrkArm
from FLSpegtransferHO.traj_opt.CubicOptimizer_1wp import CubicOptimizer_1wp
from FLSpegtransferHO.traj_opt.CubicOptimizer_1wp_ import CubicOptimizer_1wp_
import FLSpegtransferHO.motion.dvrkVariables as dvrkVar
from FLSpegtransferHO.motion.dvrkKinematics import dvrkKinematics
import FLSpegtransferHO.utils.CmnUtil as U
import numpy as np
import threading, time


class dvrkPegTransferMotionHandOver:
    """
    Motion library for peg transfer
    """
    def __init__(self, use_controller, use_optimization):
        # motion library
        self.use_optimization = use_optimization
        if use_controller:
            self.arm1 = dvrkController(arm_name='/PSM1', comp_hysteresis=True, stop_collision=False)
            self.arm2 = dvrkController(arm_name='/PSM2', comp_hysteresis=True, stop_collision=False)
        else:
            self.arm1 = dvrkArm('/PSM1')
            self.arm2 = dvrkArm('/PSM2')
        self.motion_opt_1wp_PSM1 = CubicOptimizer_1wp()
        self.motion_opt_1wp_PSM2 = CubicOptimizer_1wp_()

        # Motion variables
        self.jaw_opening = np.deg2rad([50, 50])     # PSM1, PSM2
        self.jaw_closing = np.deg2rad([-20, -15])

        self.pos_org1 = [0.060, 0.0, -0.095]
        self.rot_org1 = [0.0, 0.0, 0.0, 1.0]
        self.jaw_org1 = self.jaw_closing[0]
        self.pos_org2 = [-0.060, 0.0, -0.095]
        self.rot_org2 = [0.0, 0.0, 0.0, 1.0]
        self.jaw_org2 = self.jaw_closing[1]

        self.offset_grasp = [-0.0045, +0.003]
        self.height_ready = -0.122
        self.height_drop = -0.142
        self.height_handover = self.height_ready + 0.01
        self.height_ready_handover = self.height_handover + 0.013
        self.offset_handover = 0.01   # the amount of move to hand off

        self.lock = threading.Lock()

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
        obj.set_jaw_interpolate(jaw=[jaw])

    def move_trajectory(self, obj, q_pos):
        for joint in q_pos:
            start = time.perf_counter()
            obj.set_joint_direct(joint=joint)
            end = time.perf_counter()
            delta = 0.01 - (end-start)
            if delta > 0:
                time.sleep(delta)

    def move_trajectory2(self, obj, q_pos):
        for joint in q_pos:
            start = time.perf_counter()
            obj.set_joint_direct(joint=joint)
            end = time.perf_counter()
            delta = 0.01 - (end-start)
            if delta > 0:
                time.sleep(delta)

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
            t1 = threading.Thread(target=obj.set_pose_interpolate, args=(pos, rot))
        t2 = threading.Thread(target=obj.set_jaw_interpolate, args=[[jaw]])
        t1.daemon = True
        t2.daemon = True
        t1.start(); t2.start()
        t1.join(); t2.join()

    def handover_ready(self, which_arm='PSM1'):
        if which_arm=='PSM1':
            self.move_upright(pos=[0.125, 0.0, self.height_ready_handover],
                              rot=0.0, jaw='open', which_arm=which_arm)
        elif which_arm=='PSM2':
            self.move_upright(pos=[-0.115, 0.0, self.height_ready_handover],
                              rot=0.0, jaw='open', which_arm=which_arm)
        else:
            raise ValueError

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
        t1 = threading.Thread(target=obj.set_pose_interpolate, args=([pos[0], pos[1], pos[2]+self.offset_grasp[1]],
                                                                     U.euler_to_quaternion(np.deg2rad([rot, 0.0, 0.0]))))
        t2 = threading.Thread(target=obj.set_jaw_interpolate, args=[[self.jaw_opening[index]]])
        t1.daemon = True
        t2.daemon = True
        t1.start(); t2.start()
        t1.join(); t2.join()

        # go down toward block & close jaw
        obj.set_pose_interpolate(pos=[pos[0], pos[1], pos[2]+self.offset_grasp[0]],
                     rot=U.euler_to_quaternion(np.deg2rad([rot, 0.0, 0.0])))
        obj.set_jaw_interpolate(jaw=[self.jaw_closing[index]])

    def go_pick_traj(self, obj, obj_opt, pos_pick, rot_pick, optimized):
        self.lock.acquire()
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
            q_pos, _ = obj_opt.optimize(q0, qw, qf, dqw, max_vel=dvrkVar.v_max, max_acc=dvrkVar.a_max,
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
        self.lock.release()
        return q_pos

    def go_pick(self, pos_pick, rot_pick, which_arm):
        if which_arm == 'PSM1':
            obj = self.arm1
            obj_opt = self.motion_opt_1wp_PSM1
            index = 0
        elif which_arm == 'PSM2':
            obj = self.arm2
            obj_opt = self.motion_opt_1wp_PSM2
            index = 1
        else:
            raise ValueError
        q_pos = self.go_pick_traj(obj, obj_opt, pos_pick, rot_pick, self.use_optimization)

        # be ready to place & close jaw
        if which_arm == 'PSM1':
            t1 = threading.Thread(target=self.move_trajectory, args=(obj, q_pos))
        elif which_arm == 'PSM2':
            t1 = threading.Thread(target=self.move_trajectory2, args=(obj, q_pos))
        t2 = threading.Thread(target=obj.set_jaw_interpolate, args=[[self.jaw_closing[index]]])
        t1.daemon = True
        t2.daemon = True
        t1.start(); t2.start()
        t1.join(); t2.join()

    def go_handover_traj(self, obj_opt, pos_pick, rot_pick, pos_place, rot_place, optimized):
        self.lock.acquire()
        if optimized:
            # trajectory to transferring block from peg to peg
            q0 = dvrkKinematics.pose_to_joint(pos=[pos_pick[0], pos_pick[1], pos_pick[2] + self.offset_grasp[0]],
                                              rot=U.euler_to_quaternion(np.deg2rad([rot_pick, 0.0, 0.0])))
            qw = dvrkKinematics.pose_to_joint(pos=[pos_pick[0], pos_pick[1], self.height_ready],
                                               rot=U.euler_to_quaternion(np.deg2rad([rot_pick, 0.0, 0.0])))
            qf = dvrkKinematics.pose_to_joint(pos=[pos_place[0], pos_place[1], self.height_handover],
                                              rot=U.euler_to_quaternion(np.deg2rad([rot_place, 0.0, 0.0])))
            J = dvrkKinematics.jacobian(qw)
            dvw = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
            dqw = np.linalg.inv(J).dot(dvw)
            q_pos, _ = obj_opt.optimize(q0, qw, qf, dqw, max_vel=dvrkVar.v_max, max_acc=dvrkVar.a_max,
                                                    t_step=0.01, print_out=False, visualize=False)
        else:
            # trajectory to transferring block from peg to peg
            pos0 = [pos_pick[0], pos_pick[1], pos_pick[2] + self.offset_grasp[0]]
            rot0 = [np.deg2rad(rot_pick), 0.0, 0.0]
            pose0 = np.concatenate((pos0, rot0))

            posw = [pos_pick[0], pos_pick[1], self.height_ready]
            rotw = [np.deg2rad(rot_pick), 0.0, 0.0]
            posew = np.concatenate((posw, rotw))

            posf = [pos_place[0], pos_place[1], self.height_handover]
            rotf = [np.deg2rad(rot_place), 0.0, 0.0]
            posef = np.concatenate((posf, rotf))

            # Define trajectory
            _, q_pos1 = dvrkArm.cubic_cartesian(pose0, posew, vel_limit=dvrkVar.v_max, acc_limit=dvrkVar.a_max,
                                                tf_init=0.5, t_step=0.01)
            _, q_pos2 = dvrkArm.cubic_cartesian(posew, posef, vel_limit=dvrkVar.v_max, acc_limit=dvrkVar.a_max,
                                                tf_init=0.5, t_step=0.01)
            q_pos = np.concatenate((q_pos1, q_pos2))
        self.lock.release()
        return q_pos

    def go_handover(self, pos_pick, rot_pick, pos_place, rot_place, which_arm):
        if which_arm == 'PSM1':
            obj = self.arm1
            obj_opt = self.motion_opt_1wp_PSM1
            index = 0
        elif which_arm == 'PSM2':
            obj = self.arm2
            obj_opt = self.motion_opt_1wp_PSM2
            index = 1
        else:
            raise ValueError

        # open jaw
        obj.set_jaw_interpolate([self.jaw_opening[index]])

        # go down toward block & close jaw
        obj.set_pose_interpolate(pos=[pos_pick[0], pos_pick[1], pos_pick[2]+self.offset_grasp[0]],
                                       rot=U.euler_to_quaternion(np.deg2rad([rot_pick, 0.0, 0.0])))
        obj.set_jaw_interpolate([self.jaw_closing[index]])

        # pick-up and handover
        q_pos = self.go_handover_traj(obj_opt, pos_pick, rot_pick, pos_place, rot_place, self.use_optimization)
        if which_arm == 'PSM1':
            t1 = threading.Thread(target=self.move_trajectory, args=(obj, q_pos))
        elif which_arm == 'PSM2':
            t1 = threading.Thread(target=self.move_trajectory2, args=(obj, q_pos))
        t2 = threading.Thread(target=obj.set_jaw_interpolate, args=[[self.jaw_closing[index]]])
        t1.daemon = True
        t2.daemon = True
        t1.start(); t2.start()
        t1.join(); t2.join()

    def go_place_traj(self, obj, obj_opt, pos_place, rot_place, optimized):
        self.lock.acquire()
        if optimized:
            # trajectory to pick
            q0 = obj.get_current_joint()
            qw = dvrkKinematics.pose_to_joint(pos=[pos_place[0], pos_place[1], self.height_ready],
                                              rot=U.euler_to_quaternion(np.deg2rad([rot_place, 0.0, 0.0])))
            qf = dvrkKinematics.pose_to_joint(pos=[pos_place[0], pos_place[1], self.height_drop],
                                              rot=U.euler_to_quaternion(np.deg2rad([rot_place, 0.0, 0.0])))
            J = dvrkKinematics.jacobian(qw)
            dvw = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
            dqw = np.linalg.inv(J).dot(dvw)
            q_pos, _ = obj_opt.optimize(q0, qw, qf, dqw, max_vel=dvrkVar.v_max, max_acc=dvrkVar.a_max,
                                                    t_step=0.01, print_out=False, visualize=False)
        else:
            pos0, quat0 = obj.get_current_pose()
            rot0 = U.quaternion_to_euler(quat0)
            pose0 = np.concatenate((pos0, rot0))

            posw = [pos_place[0], pos_place[1], self.height_ready]
            rotw = [np.deg2rad(rot_place), 0.0, 0.0]
            posew = np.concatenate((posw, rotw))

            posf = [pos_place[0], pos_place[1], self.height_drop]
            rotf = [np.deg2rad(rot_place), 0.0, 0.0]
            posef = np.concatenate((posf, rotf))

            # Define trajectory
            _, q_pos1 = dvrkArm.cubic_cartesian(pose0, posew, vel_limit=dvrkVar.v_max, acc_limit=dvrkVar.a_max,
                                                tf_init=0.5, t_step=0.01)
            _, q_pos2 = dvrkArm.cubic_cartesian(posew, posef, vel_limit=dvrkVar.v_max, acc_limit=dvrkVar.a_max,
                                                tf_init=0.5, t_step=0.01)
            q_pos = np.concatenate((q_pos1, q_pos2))
        self.lock.release()
        return q_pos

    def go_place(self, pos_place, rot_place, which_arm):
        if which_arm == 'PSM1':
            obj = self.arm1
            obj_opt = self.motion_opt_1wp_PSM1
            index = 0
        elif which_arm == 'PSM2':
            obj = self.arm2
            obj_opt = self.motion_opt_1wp_PSM2
            index = 1
        else:
            raise ValueError

        q_pos = self.go_place_traj(obj, obj_opt, pos_place, rot_place, self.use_optimization)

        # be ready to place & close jaw
        if which_arm == 'PSM1':
            self.move_trajectory(obj, q_pos)
        elif which_arm == 'PSM2':
            self.move_trajectory2(obj, q_pos)
        obj.set_jaw_interpolate([self.jaw_opening[index]])


if __name__ == "__main__":
    motion = dvrkPegTransferMotionHandOver()