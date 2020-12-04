from FLSpegtransfer.vision.BallDetectionRGBD import BallDetectionRGBD
from FLSpegtransfer.motion.dvrkDualArm import dvrkDualArm
from FLSpegtransfer.vision.AlliedVisionCapture import AlliedVisionCapture
from FLSpegtransfer.vision.AlliedVisionUtils import AlliedVisionUtils
from FLSpegtransfer.motion.dvrkKinematics import dvrkKinematics
from FLSpegtransfer.vision.ToolDetectionStereo import ToolDetection
from FLSpegtransfer.vision.NeedleTrackingStereo import NeedleTrackingStereoEllipse
import FLSpegtransfer.motion.dvrkVariables as dvrkVar
import FLSpegtransfer.utils.CmnUtil as U
from FLSpegtransfer.utils.ImgUtils import ImgUtils
from FLSpegtransfer.path import *
import numpy as np
import time
import cv2

# define instances
# av = AlliedVisionCapture()
av_util = AlliedVisionUtils()
# dvrk = dvrkDualArm()
Trc = np.load(root + 'calibration_files/Trc_stereo_PSM1.npy')
td = ToolDetection(Trc=Trc)
nd = NeedleTrackingStereoEllipse()

# q_init = [0.71, -0.163, 0.150, 0.133, -0.107, 0.905]
# dvrk.set_joint(joint1=q_init)
# dvrk.set_jaw(jaw1=[np.deg2rad(20)])
cnt = 0
while True:
    # img_left = cv2.imread('img_left' + str(cnt) + '.png')
    # img_right = cv2.imread('img_right' + str(cnt) + '.png')
    img_left = np.load('record_needle_manipulation/img_left' + str(cnt) + '.npy')
    img_right = np.load('record_needle_manipulation/img_right' + str(cnt) + '.npy')

    # img_left, img_right = av.capture(which='rectified')
    if len(img_left) == 0 or len(img_right) == 0:
        pass
    else:
        # Find needle
        nd.find_needle3D(img_left, img_right, color='blue', visualize=False)
        if cnt == 397:
        # if cnt == 50:
            cnt = 0
        else:
            cnt += 1

        # Pose estimation
        # td.find_tool3D(img_left, img_right, color='red', visualize=False)

        # pt, q_phy = pose_estimation(pbr, pbg, pbb, pby, use_Trc=True)

        # # visual servoing controller
        # q_err = (np.array(q_des) - np.array(q_phy))
        # q_crit = np.deg2rad(0.5)    # break criteria
        # if abs(q_err[4]) < q_crit and abs(q_err[5]) < q_crit:
        #     print (step)
        #     np.save("q_des", q_des_)
        #     np.save("q_phy", q_phy_)
        #     np.save("q_cmd", q_cmd_)
        #     np.save("q_err", q_err_)
        #     break
        # else:
        #     q_cmd[4] += q_err[4]*0.3
        #     q_cmd[5] += q_err[5]*0.3
        #     # dvrk.set_jaw(jaw1=[np.deg2rad(5)])
        #     # dvrk.set_joint(joint1=q_cmd)
        #     step += 1
        #
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
