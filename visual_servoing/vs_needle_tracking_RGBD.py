from FLSpegtransfer.vision.BallDetectionRGBD import BallDetectionRGBD
from FLSpegtransfer.motion.dvrkDualArm import dvrkDualArm
from FLSpegtransfer.vision.ZividCapture import ZividCapture
from FLSpegtransfer.vision.ZividUtils import ZividUtils
from FLSpegtransfer.motion.dvrkKinematics import dvrkKinematics
from FLSpegtransfer.vision.ToolDetectionRGBD import ToolDetectionRGBD
from FLSpegtransfer.vision.NeedleDetectionRGBD import NeedleDetectionRGBD
from FLSpegtransfer.utils.CmnUtil import *
import FLSpegtransfer.motion.dvrkVariables as dvrkVar
import FLSpegtransfer.utils.CmnUtil as U
from FLSpegtransfer.utils.ImgUtils import ImgUtils
from FLSpegtransfer.path import *
import numpy as np
import time
import cv2


def pose_estimation(pbr, pbg, pbb, pby, use_Trc):    # Find tool position, joint angles
    pt = []
    q_phy = []
    if len(pbr) < 2:
        pass
    else:
        pt = bd.find_tool_pitch(pbr[0], pbr[1])  # tool position of pitch axis
        pt = np.array(pt) * 0.001  # (m)
        if use_Trc:
            pt = bd.Rrc.dot(pt) + bd.trc
            qp1, qp2, qp3 = dvrkKinematics.ik_position_straight(pt, L3=0, L4=0)     # position of pitch axis

            # Find tool orientation, joint angles, and overlay
            temp = [pbr[2], pbg, pbb, pby]
            if len(pbr) < 3:
                qp4 = 0.0; qp5 = 0.0; qp6 = 0.0
            elif temp.count([]) > 2:
                qp4=0.0; qp5=0.0; qp6=0.0
            else:
                Rm = bd.find_tool_orientation(pbr[2], pbg, pbb, pby)  # orientation of the marker
                qp4, qp5, qp6 = dvrkKinematics.ik_orientation(qp1, qp2, Rm)
            q_phy = [qp1, qp2, qp3, qp4, qp5, qp6]
        else:
            q_phy = []
    return pt, q_phy

# define instances
zivid = ZividCapture(which_camera='inclined')
zivid_util = ZividUtils(which_camera='inclined')
zivid.start()

dvrk = dvrkDualArm()
Trc = np.load(root + 'calibration_files/Trc_inclined1.npy')
td = ToolDetectionRGBD(Trc=Trc)
nd = NeedleDetectionRGBD()

# q_init = [0.71, -0.163, 0.150, 0.133, -0.107, 0.905]
# dvrk.set_joint(joint1=q_init)
rot = [-90.0, 0.0, 0.0]
quat = euler_to_quaternion(rot, unit='deg')
dvrk.set_pose(rot1=quat)
dvrk.set_jaw(jaw1=[np.deg2rad(-5)])
while True:
    img_color, img_depth, img_point = zivid.capture_3Dimage(color='BGR')

    # Find needle
    pn = nd.find_needle(img_color, img_point, color='blue', visualize=False)
    pt = td.find_tool(img_color, img_point, color='red', visualize=False)
    pn_robot = U.transform(pn*0.001, Trc)
    pn_robot_above = pn_robot + np.array([0.0, 0.0, 0.005])
    pt_robot = U.transform(pt*0.001, Trc)

    # visual servoing controller
    q_err = (np.array(q_des) - np.array(q_phy))
    q_crit = np.deg2rad(0.5)    # break criteria
    if abs(q_err[4]) < q_crit and abs(q_err[5]) < q_crit:
        print (step)
        np.save("q_des", q_des_)
        np.save("q_phy", q_phy_)
        np.save("q_cmd", q_cmd_)
        np.save("q_err", q_err_)
        break
    else:
        q_cmd[4] += q_err[4]*0.3
        q_cmd[5] += q_err[5]*0.3
        # dvrk.set_jaw(jaw1=[np.deg2rad(5)])
        # dvrk.set_joint(joint1=q_cmd)
        step += 1

    # print ("q_des: ", q_des[4:])
    # print ("q_phy: ", q_phy[4:])
    # print ("q_cmd: ", q_cmd[4:])
    # print ("q_err: ", q_err[4:])
    # print ("")
    # q_des_.append(q_des[4:])
    # q_phy_.append(q_phy[4:])
    # q_cmd_.append(q_cmd[4:])
    # q_err_.append(q_err[4:])
    #
    # # Visualize
    # color = bd.overlay_ball(color, pbr)
    # color = bd.overlay_ball(color, [pbg])
    # color = bd.overlay_ball(color, [pbb])
    # color = bd.overlay_ball(color, [pby])
    # color = bd.overlay_tool(color, q_phy, (0, 255, 0))
    # cv2.imshow("images", color)
    # cv2.waitKey(1) & 0xFF
