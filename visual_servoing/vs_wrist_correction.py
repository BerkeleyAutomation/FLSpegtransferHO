from FLSpegtransfer.vision.BallDetectionRGBD import BallDetectionRGBD
from FLSpegtransfer.motion.dvrkDualArm import dvrkDualArm
from FLSpegtransfer.vision.ZividCapture import ZividCapture
from FLSpegtransfer.motion.dvrkKinematics import dvrkKinematics
import FLSpegtransfer.motion.dvrkVariables as dvrkVar
import FLSpegtransfer.utils.CmnUtil as U
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
which_camera = 'inclined'
zivid = ZividCapture(which_camera=which_camera)
zivid.start()
dvrk = dvrkDualArm()
Trc = np.load(root+'calibration_files/Trc_' + which_camera + '.npy')
Tpc = np.load(root+'calibration_files/Tpc_' + which_camera + '.npy')
bd = BallDetectionRGBD(Trc=Trc, Tpc=Tpc, use_rc_transform=True, which_camera=which_camera)

# for data record
q_des_ = []
q_phy_ = []
q_cmd_ = []
q_err_ = []

q_init = [0.88, 0.045, 0.191, 0.0899, 0.229, -0.815]
dvrk.set_jaw(jaw1=[np.deg2rad(5)])
dvrk.set_joint(joint1=q_init)
q_des = [0.88, 0.045, 0.191, 0.0899, 0.229, -0.815]
q_des[4] = 0.5
q_des[5] = -0.3
q_cmd = q_des.copy()
step = 0
while True:
    # Capture image from Zivid
    color, _, point = zivid.capture_3Dimage(color='BGR')
    color_org = np.copy(color)

    # Find balls
    pbr = bd.find_balls(color, point, 'red', nb_sphere=3, visualize=False)
    pbg = bd.find_balls(color, point, 'green', nb_sphere=1, visualize=False)
    pbb = bd.find_balls(color, point, 'blue', nb_sphere=1, visualize=False)
    pby = bd.find_balls(color, point, 'yellow', nb_sphere=1, visualize=False)

    # Pose estimation
    pt, q_phy = pose_estimation(pbr, pbg, pbb, pby, use_Trc=True)

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

    print ("q_des: ", q_des[4:])
    print ("q_phy: ", q_phy[4:])
    print ("q_cmd: ", q_cmd[4:])
    print ("q_err: ", q_err[4:])
    print ("")
    q_des_.append(q_des[4:])
    q_phy_.append(q_phy[4:])
    q_cmd_.append(q_cmd[4:])
    q_err_.append(q_err[4:])

    # Visualize
    color = bd.overlay_ball(color, pbr)
    color = bd.overlay_ball(color, [pbg])
    color = bd.overlay_ball(color, [pbb])
    color = bd.overlay_ball(color, [pby])
    color = bd.overlay_tool(color, q_phy, (0, 255, 0))
    cv2.imshow("images", color)
    cv2.waitKey(1) & 0xFF
