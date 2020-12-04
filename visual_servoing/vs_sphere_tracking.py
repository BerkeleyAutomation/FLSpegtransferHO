from FLSpegtransfer.vision.AlliedVisionCapture import AlliedVisionCapture
from FLSpegtransfer.vision.BallDetectionStereo import BallDetectionStereo
from FLSpegtransfer.vision.ToolDetectionStereo import ToolDetection
from FLSpegtransfer.motion.dvrkDualArm import dvrkDualArm
from FLSpegtransfer.utils.ImgUtils import ImgUtils
import time
import numpy as np
import cv2

# define instances
bd = BallDetectionStereo()
td = ToolDetection()
av = AlliedVisionCapture()
dvrk = dvrkDualArm()

while True:
    img_left, img_right = av.capture(which='rectified')
    if len(img_left) == 0 or len(img_right) == 0:
        pass
    else:
        # img_compared = compare_rectified_img(img_rect1, img_rect2, scale=0.7, line_gap=40)
        # cv2.imshow("stacked", img_compared)
        # cv2.waitKey(0)

        # Find balls
        pbr = np.array(td.find_tool3D(img_left, img_right, color='red', visualize=False))
        pbg = np.array(bd.find_balls3D(img_left, img_right, color='green', visualize=False))

        if len(pbg) == 0 or len(pbr) == 0:
            pass
        else:
            # Visualize
            img_left = bd.overlay_ball(img_left, pbg, which='left')
            img_left = td.overlay_tooltip(img_left, pbr, which='left')
            img_left = bd.overlay_vector(img_left, [pbr], pbg, which='left')
            img_right = bd.overlay_ball(img_right, pbg, which='right')
            img_right = td.overlay_tooltip(img_right, pbr, which='right')
            img_right = bd.overlay_vector(img_right, [pbr], pbg, which='right')
            img_stacked = ImgUtils.stack_stereo_img(img_left, img_right, scale=0.7)
            cv2.imshow("", img_stacked)
            cv2.waitKey(1)

            q_des = td.transform(pbg[0][:3], td.Trc)
            q_act = td.transform(pbr[:3], td.Trc)
            error = q_des - q_act
            norm = np.linalg.norm(error)
            print (error, norm)

            pose1, _ = dvrk.get_pose()
            pos1 = pose1[0]
            pos1_corrected = pos1 + error/norm*0.003
            if norm > 0.01:
                dvrk.set_arm_position(pos1=pos1_corrected, wait_callback=False)
            time.sleep(0.001)