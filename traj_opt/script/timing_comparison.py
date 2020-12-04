from FLSpegtransfer.vision.BlockDetection3D import BlockDetection3D
from FLSpegtransfer.vision.GraspingPose3D import GraspingPose3D
from FLSpegtransfer.vision.VisualizeDetection import VisualizeDetection
from FLSpegtransfer.vision.PegboardCalibration import PegboardCalibration
from FLSpegtransfer.motion.dvrkKinematics import dvrkKinematics
from FLSpegtransfer.path import *
from FLSpegtransfer.traj_opt.PegMotionOptimizer_2wp import PegMotionOptimizer_2wp
import FLSpegtransfer.utils.CmnUtil as U
import numpy as np
from FLSpegtransfer.motion.dvrkArm import dvrkArm
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


# trajectory to transferring block from peg to peg
def transfer_block(optimizer, pos_pick, rot_pick, pos_place, rot_place, max_vel, max_acc, method):
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
                                                max_vel=max_vel, max_acc=max_acc, t_step=0.01,
                                                horizon=50, print_out=True, visualize=False)
    elif method=='LSPB_joint':
        q_pos = []
        t1, traj1 = dvrkArm.LSPB(q0, qw1, v_max=max_vel, a_max=max_acc)
        t2, traj2 = dvrkArm.LSPB(qw1, qw2, v_max=max_vel, a_max=max_acc)
        t3, traj3 = dvrkArm.LSPB(qw2, qf, v_max=max_vel, a_max=max_acc)
        q_pos.append(traj1)
        q_pos.append(traj2)
        q_pos.append(traj3)
    elif method=='cubic_cartesian':
        raise NotImplementedError
        pose0 = [pos_pick[0], pos_pick[1], pos_pick[2] + offset_grasp[0]] + [np.deg2rad(rot_pick)]  # [x,y,z,ang]
        posew1 = [pos_pick[0], pos_pick[1], height_ready] + [np.deg2rad(rot_pick)]
        posew2 = [pos_place[0], pos_place[1], height_ready] + [np.deg2rad(rot_place)]
        posef = [pos_place[0], pos_place[1], height_drop] + [np.deg2rad(rot_place)]
        # q_pos = []
        # t1, traj1 = dvrkArm.cubic_time(pose0, posew1, tf=time)
        # t2, traj2 = dvrkArm.cubic_time(posew1, posew2, tf=time)
        # t3, traj3 = dvrkArm.cubic_time(posew2, posef, tf=time)
    else:
        raise ValueError
    return q_pos


# trajectory of returning to another peg to pick-up
def return_to_peg(optimizer, pos_place_curr, rot_place_curr, pos_pick_next, rot_pick_next, max_vel, max_acc, method):
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
                                      max_vel=max_vel, max_acc=max_acc, t_step=0.01,
                                      horizon=50, print_out=True, visualize=False)
    elif method=='LSPB_joint':
        q_pos = []
        t1, traj1 = dvrkArm.LSPB(q0, qw1, v_max=max_vel, a_max=max_acc)
        t2, traj2 = dvrkArm.LSPB(qw1, qw2, v_max=max_vel, a_max=max_acc)
        t3, traj3 = dvrkArm.LSPB(qw2, qf, v_max=max_vel, a_max=max_acc)
        q_pos.append(traj1)
        q_pos.append(traj2)
        q_pos.append(traj3)
    elif method=='cubic_cartesian':
        raise NotImplementedError
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

# action ordering
action_list = np.array([[1, 7], [0, 6], [3, 8], [2, 9], [5, 11], [4, 10],  # left to right
                        [7, 1], [6, 0], [8, 3], [9, 2], [11, 5], [10, 4]])  # right to left

# define constraints
max_vel = [1.0, 1.0, 0.2, 8.0, 8.0, 8.0]       # max velocity (rad/s) or (m/s)
max_acc = [1.0, 1.0, 0.2, 8.0, 8.0, 8.0]     # max acceleration (rad/s^2) or (m/s^2)

traj = []
traj_opt = []
for action_order, action in enumerate(action_list):
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
        qs = return_to_peg(optimizer=motion_opt_2wp, pos_place_curr=gp_place_robot_prev, rot_place_curr=gp_place_prev[0],
                           pos_pick_next=gp_pick_robot, rot_pick_next=gp_pick[0], max_vel=max_vel, max_acc=max_acc, method='cubic_cartesian')
        traj.append(qs)
        qs = return_to_peg(optimizer=motion_opt_2wp, pos_place_curr=gp_place_robot_prev, rot_place_curr=gp_place_prev[0],
                           pos_pick_next=gp_pick_robot, rot_pick_next=gp_pick[0], max_vel=max_vel, max_acc=max_acc, method='optimization')
        traj_opt.append(qs)
    qs = transfer_block(optimizer=motion_opt_2wp, pos_pick=gp_pick_robot, rot_pick=gp_pick[0],
                          pos_place=gp_place_robot, rot_place=gp_place[0], max_vel=max_vel, max_acc=max_acc, method='cubic_cartesian')
    traj.append(qs)
    qs = transfer_block(optimizer=motion_opt_2wp, pos_pick=gp_pick_robot, rot_pick=gp_pick[0],
                          pos_place=gp_place_robot, rot_place=gp_place[0], max_vel=max_vel, max_acc=max_acc, method='optimization')
    traj_opt.append(qs)
    gp_place_prev = gp_place
    gp_place_robot_prev = gp_place_robot

# # plot
# for qs, qs_opt in zip(traj, traj_opt):
#     qs = np.concatenate(qs)
#     print (len(qs), len(qs_opt))
#
# # run motion
# dvrk = dvrkArm('/PSM1')
# for tt in traj_opt:
#     # for qs in tt:
#     dvrk.set_joint(tt[0], wait_callback=True)
#     for q in tt:
#         dvrk.set_joint(q, wait_callback=False)
#         time.sleep(0.01)
#     dvrk.set_joint(tt[-1], wait_callback=True)