from FLSpegtransfer.vision.BlockDetection3D import BlockDetection3D
from FLSpegtransfer.vision.GraspingPose3D import GraspingPose3D
from FLSpegtransfer.vision.VisualizeDetection import VisualizeDetection
from FLSpegtransfer.vision.PegboardCalibration import PegboardCalibration
from FLSpegtransfer.motion.dvrkKinematics import dvrkKinematics
from FLSpegtransfer.path import *
from FLSpegtransfer.traj_opt.PegMotionOptimizer_2wp import PegMotionOptimizer_2wp
from FLSpegtransfer.traj_opt.CubicOptimizer_2wp import CubicOptimizer_2wp
from FLSpegtransfer.motion.dvrkArm import dvrkArm
import FLSpegtransfer.utils.CmnUtil as U
import numpy as np
import time


def transform_task2robot(point, Trc, Tpc):
    point = np.array(point)
    Tcp = np.linalg.inv(Tpc)  # (mm)
    Tcp[:3, -1] = Tcp[:3, -1] * 0.001  # (m)
    Trp = Trc.dot(Tcp)
    Rrp = Trp[:3, :3]
    trp = Trp[:3, -1]
    transformed = Rrp.dot(point * 0.001) + trp
    return transformed


def cubic_cartesian(pose0, posef, vel_limit, acc_limit, tf_init):
    tf = tf_init
    while True:
        _, pose_traj = dvrkArm.cubic_time(pose0, posef, tf=tf, t_step=0.01)

        # pose_traj in cartesian to joints_traj
        dt = 0.01
        q_pos = np.array([dvrk_model.pose_to_joint(conf[:3], rot=U.euler_to_quaternion([conf[3], 0.0, 0.0])) for conf in pose_traj])
        q_pos_prev = np.insert(q_pos, 0, q_pos[0], axis=0)
        q_pos_prev = np.delete(q_pos_prev, -1, axis=0)
        q_vel = (q_pos - q_pos_prev) / dt
        q_vel_prev = np.insert(q_vel, 0, q_vel[0], axis=0)
        q_vel_prev = np.delete(q_vel_prev, -1, axis=0)
        q_acc = (q_vel - q_vel_prev) / dt

        # find maximum values
        vel_max = np.max(abs(q_vel), axis=0)
        acc_max = np.max(abs(q_acc), axis=0)
        if np.any(vel_max > vel_limit) or np.any(acc_max > acc_limit):
            break
        else:
            tf += -0.01
    return tf, q_pos


# trajectory to transferring block from peg to peg
def transfer_block(optimizer, pos_pick, rot_pick, pos_place, rot_place, vel_limit, acc_limit, method):
    offset_grasp = [-0.005, +0.002]
    height_ready = -0.120
    height_drop = -0.15
    q0 = dvrk_model.pose_to_joint(pos=[pos_pick[0], pos_pick[1], pos_pick[2] + offset_grasp[0]],
                                  rot=U.euler_to_quaternion(np.deg2rad([rot_pick, 0.0, 0.0])))
    qw1 = dvrk_model.pose_to_joint(pos=[pos_pick[0], pos_pick[1], height_ready],
                                   rot=U.euler_to_quaternion(np.deg2rad([rot_pick, 0.0, 0.0])))
    qw2 = dvrk_model.pose_to_joint(pos=[pos_place[0], pos_place[1], height_ready],
                                   rot=U.euler_to_quaternion(np.deg2rad([rot_place, 0.0, 0.0])))
    qf = dvrk_model.pose_to_joint(pos=[pos_place[0], pos_place[1], height_drop],
                                  rot=U.euler_to_quaternion(np.deg2rad([rot_place, 0.0, 0.0])))
    if method=='optimization':
        J1 = dvrk_model.jacobian(qw1)
        J2 = dvrk_model.jacobian(qw2)
        dvw = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        dqw1 = np.linalg.inv(J1).dot(dvw)
        dqw2 = np.linalg.inv(J2).dot(dvw)
        q_pos, q_vel, q_acc, t =\
            optimizer.optimize_motion(q0, qw1, dqw1, qw2, dqw2, qf,
                                                max_vel=vel_limit, max_acc=acc_limit, t_step=0.01,
                                                horizon=50, print_out=True, visualize=False)
    elif method=='LSPB_joint':
        q_pos = []
        t1, traj1 = dvrkArm.LSPB(q0, qw1, v_max=vel_limit, a_max=acc_limit)
        t2, traj2 = dvrkArm.LSPB(qw1, qw2, v_max=vel_limit, a_max=acc_limit)
        t3, traj3 = dvrkArm.LSPB(qw2, qf, v_max=vel_limit, a_max=acc_limit)
        q_pos.append(traj1)
        q_pos.append(traj2)
        q_pos.append(traj3)
    elif method=='cubic_optimizer':
        J1 = dvrk_model.jacobian(qw1)
        J2 = dvrk_model.jacobian(qw2)
        dvw = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        dqw1 = np.linalg.inv(J1).dot(dvw)
        dqw2 = np.linalg.inv(J2).dot(dvw)
        q_pos, t = optimizer.optimize(q0, qw1, qw2, qf, dqw1, dqw2, max_vel=vel_limit, max_acc=acc_limit, t_step=0.01,
                                      print_out=False, visualize=False)
    elif method=='cubic_cartesian':
        pose0 = [pos_pick[0], pos_pick[1], pos_pick[2] + offset_grasp[0]] + [np.deg2rad(rot_pick)]  # [x,y,z,ang]
        posew1 = [pos_pick[0], pos_pick[1], height_ready] + [np.deg2rad(rot_pick)]
        posew2 = [pos_place[0], pos_place[1], height_ready] + [np.deg2rad(rot_place)]
        posef = [pos_place[0], pos_place[1], height_drop] + [np.deg2rad(rot_place)]
        q_pos = []
        tf1, traj1 = cubic_cartesian(pose0, posew1, vel_limit=vel_limit, acc_limit=acc_limit, tf_init=2.0)
        tf2, traj2 = cubic_cartesian(posew1, posew2, vel_limit=vel_limit, acc_limit=acc_limit, tf_init=2.0)
        tf3, traj3 = cubic_cartesian(posew2, posef, vel_limit=vel_limit, acc_limit=acc_limit, tf_init=2.0)
        q_pos.append(traj1)
        q_pos.append(traj2)
        q_pos.append(traj3)
    else:
        raise ValueError
    return q_pos


# trajectory of returning to another peg to pick-up
def return_to_peg(optimizer, pos_place_curr, rot_place_curr, pos_pick_next, rot_pick_next, vel_limit, acc_limit, method):
    offset_grasp = [-0.005, +0.002]
    height_ready = -0.120
    height_drop = -0.15
    q0 = dvrk_model.pose_to_joint(pos=[pos_place_curr[0], pos_place_curr[1], height_drop],
                                  rot=U.euler_to_quaternion(np.deg2rad([rot_place_curr, 0.0, 0.0])))
    qw1 = dvrk_model.pose_to_joint(pos=[pos_place_curr[0], pos_place_curr[1], height_ready],
                                   rot=U.euler_to_quaternion(np.deg2rad([rot_place_curr, 0.0, 0.0])))
    qw2 = dvrk_model.pose_to_joint(pos=[pos_pick_next[0], pos_pick_next[1], height_ready],
                                   rot=U.euler_to_quaternion(np.deg2rad([rot_pick_next, 0.0, 0.0])))
    qf = dvrk_model.pose_to_joint(pos=[pos_pick_next[0], pos_pick_next[1], pos_pick_next[2] + offset_grasp[0]],
                                  rot=U.euler_to_quaternion(np.deg2rad([rot_pick_next, 0.0, 0.0])))
    if method=='optimization':
        J1 = dvrk_model.jacobian(qw1)
        J2 = dvrk_model.jacobian(qw2)
        dvw = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        dqw1 = np.linalg.inv(J1).dot(dvw)
        dqw2 = np.linalg.inv(J2).dot(dvw)
        q_pos, q_vel, q_acc, t = \
            optimizer.optimize_motion(q0, qw1, dqw1, qw2, dqw2, qf,
                                      max_vel=vel_limit, max_acc=acc_limit, t_step=0.01,
                                      horizon=50, print_out=True, visualize=False)
    elif method=='LSPB_joint':
        q_pos = []
        t1, traj1 = dvrkArm.LSPB(q0, qw1, v_max=vel_limit, a_max=acc_limit)
        t2, traj2 = dvrkArm.LSPB(qw1, qw2, v_max=vel_limit, a_max=acc_limit)
        t3, traj3 = dvrkArm.LSPB(qw2, qf, v_max=vel_limit, a_max=acc_limit)
        q_pos.append(traj1)
        q_pos.append(traj2)
        q_pos.append(traj3)
    elif method=='cubic_optimizer':
        J1 = dvrk_model.jacobian(qw1)
        J2 = dvrk_model.jacobian(qw2)
        dvw = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        dqw1 = np.linalg.inv(J1).dot(dvw)
        dqw2 = np.linalg.inv(J2).dot(dvw)
        q_pos, t =\
            optimizer.optimize(q0, qw1, qw2, qf, dqw1, dqw2, max_vel=vel_limit, max_acc=acc_limit, t_step=0.01,
                               print_out=True, visualize=False)
    elif method=='cubic_cartesian':
        pose0 = [pos_place_curr[0], pos_place_curr[1], height_drop] + [np.deg2rad(rot_place_curr)]  # [x,y,z,ang]
        posew1 = [pos_place_curr[0], pos_place_curr[1], height_ready] + [np.deg2rad(rot_place_curr)]
        posew2 = [pos_pick_next[0], pos_pick_next[1], height_ready] + [np.deg2rad(rot_pick_next)]
        posef = [pos_pick_next[0], pos_pick_next[1], pos_pick_next[2] + offset_grasp[0]] + [np.deg2rad(rot_pick_next)]
        q_pos = []
        tf1, traj1 = cubic_cartesian(pose0, posew1, vel_limit=vel_limit, acc_limit=acc_limit, tf_init=2.0)
        tf2, traj2 = cubic_cartesian(posew1, posew2, vel_limit=vel_limit, acc_limit=acc_limit, tf_init=2.0)
        tf3, traj3 = cubic_cartesian(posew2, posef, vel_limit=vel_limit, acc_limit=acc_limit, tf_init=2.0)
        q_pos.append(traj1)
        q_pos.append(traj2)
        q_pos.append(traj3)
    else:
        raise ValueError
    return q_pos


# load transform
which_camera = 'inclined'
Trc = np.load(root + 'calibration_files/Trc_' + which_camera + '_PSM1.npy')  # robot to camera
Tpc = np.load(root + 'calibration_files/Tpc_' + which_camera + '.npy')  # pegboard to camera

block = BlockDetection3D(Tpc)
gp = GraspingPose3D()
vd = VisualizeDetection()
pegboard = PegboardCalibration()
dvrk_model = dvrkKinematics()
motion_opt_2wp = PegMotionOptimizer_2wp()
motion_opt_cubic = CubicOptimizer()

# action ordering
action_list = np.array([[1, 7], [0, 6], [3, 8], [2, 9], [5, 11], [4, 10],  # left to right
                        [7, 1], [6, 0], [8, 3], [9, 2], [11, 5], [10, 4]])  # right to left

# define constraints
max_vel = [1.0, 1.0, 1.0, 8.0, 8.0, 8.0]     # max velocity (rad/s) or (m/s)
max_acc = [1.0, 1.0, 1.0, 8.0, 8.0, 8.0]     # max acceleration (rad/s^2) or (m/s^2)

traj = []
traj_opt_cubic = []
traj_opt_QP = []
for action_order, action in enumerate(action_list):
    print (action)
    # load images
    if action_order <= 5:
        img_color = np.load(root + 'record/peg_transfer_kit_capture/img_color_inclined.npy')
        img_point = np.load(root + 'record/peg_transfer_kit_capture/img_point_inclined.npy')
    else:
        img_color = np.load(root + 'record/peg_transfer_kit_capture/img_color_inclined_rhp.npy')
        img_point = np.load(root + 'record/peg_transfer_kit_capture/img_point_inclined_rhp.npy')

    # find pegs
    block.find_pegs(img_color=img_color, img_point=img_point)

    # find numbering of pick & place
    nb_pick = action_list[action_order][0]
    nb_place = action_list[action_order][1]
    pose_blk_pick, pnt_blk_pick, pnt_mask_pick = block.find_block(block_number=nb_pick, img_color=img_color,
                                                                  img_point=img_point)
    pose_blk_place, pnt_blk_place, pnt_mask_place = block.find_block(block_number=nb_place, img_color=img_color,
                                                                     img_point=img_point)

    # find grasping & placing pose
    gp.find_grasping_pose(pose_blk=pose_blk_pick, peg_point=block.pnt_pegs[nb_pick])
    gp.find_placing_pose(peg_point=block.pnt_pegs[nb_place])
    gp_pick = gp.pose_grasping
    gp_place = gp.pose_placing
    gp_pick_robot = transform_task2robot(gp_pick[1:], Trc=Trc, Tpc=Tpc)  # [x,y,z]
    gp_place_robot = transform_task2robot(gp_place[1:], Trc=Trc, Tpc=Tpc)
    if gp_pick[0] == 0.0:  gp_pick[0] = 0.001     # to prevent optimizer error
    if gp_place[0] == 0.0: gp_place[0] = 0.001
    if action_order==0 or action_order==6:
        pass
    else:
        qs = return_to_peg(optimizer=motion_opt_cubic, pos_place_curr=gp_place_robot_prev, rot_place_curr=gp_place_prev[0],
                           pos_pick_next=gp_pick_robot, rot_pick_next=gp_pick[0], vel_limit=max_vel, acc_limit=max_acc, method='cubic_optimizer')
        traj_opt_cubic.append(qs)
        qs = return_to_peg(optimizer=motion_opt_2wp, pos_place_curr=gp_place_robot_prev, rot_place_curr=gp_place_prev[0],
                           pos_pick_next=gp_pick_robot, rot_pick_next=gp_pick[0], vel_limit=max_vel, acc_limit=max_acc, method='cubic_cartesian')
        traj.append(qs)
        qs = return_to_peg(optimizer=motion_opt_2wp, pos_place_curr=gp_place_robot_prev, rot_place_curr=gp_place_prev[0],
                           pos_pick_next=gp_pick_robot, rot_pick_next=gp_pick[0], vel_limit=max_vel, acc_limit=max_acc, method='optimization')
        traj_opt_QP.append(qs)

    qs = transfer_block(optimizer=motion_opt_cubic, pos_pick=gp_pick_robot, rot_pick=gp_pick[0],
                          pos_place=gp_place_robot, rot_place=gp_place[0], vel_limit=max_vel, acc_limit=max_acc, method='cubic_optimizer')
    traj_opt_cubic.append(qs)
    qs = transfer_block(optimizer=motion_opt_2wp, pos_pick=gp_pick_robot, rot_pick=gp_pick[0],
                          pos_place=gp_place_robot, rot_place=gp_place[0], vel_limit=max_vel, acc_limit=max_acc, method='cubic_cartesian')
    traj.append(qs)
    qs = transfer_block(optimizer=motion_opt_2wp, pos_pick=gp_pick_robot, rot_pick=gp_pick[0],
                          pos_place=gp_place_robot, rot_place=gp_place[0], vel_limit=max_vel, acc_limit=max_acc, method='optimization')
    traj_opt_QP.append(qs)
    gp_place_prev = gp_place
    gp_place_robot_prev = gp_place_robot

np.save("traj", traj)
np.save("traj_opt_cubic", traj_opt_cubic)
np.save("traj_opt_QP", traj_opt_QP)