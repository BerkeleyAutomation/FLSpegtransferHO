import cv2
import numpy as np
# from FLSpegtransfer.vision.Laparoscope import Laparoscope
from FLSpegtransfer.vision.BallDetectionStereo import BallDetectionStereo
from FLSpegtransfer.vision.AlliedVisionCapture import AlliedVisionCapture
from FLSpegtransfer.motion.dvrkKinematics import dvrkKinematics

BD = BallDetectionStereo()
av = AlliedVisionCapture()
# laparo = Laparoscope()

while True:
    # img_left = laparo.img_left
    # img_right = laparo.img_right
    img_left, img_right = av.capture(which='rectified')
    if len(img_left) == 0 or len(img_right) == 0:
        pass
    else:
        # img_compared = compare_rectified_img(img_rect1, img_rect2, scale=0.7, line_gap=40)
        # cv2.imshow("stacked", img_compared)
        # cv2.waitKey(0)

        # Find balls
        pbr = np.array(BD.find_balls3D(img_left, img_right, color='red', visualize=False))
        # pbg = BD.find_balls3D(img_left, img_right, 'green', visualize=False)
        # pbb = BD.find_balls3D(img_left, img_right, 'blue', visualize=False)
        # pby = BD.find_balls3D(img_left, img_right, 'yellow', visualize=False)

        # Visualize
        color = BD.overlay_ball(img_left, pbr)
        # color = BD.overlay_ball(img_left, [pbg])
        # color = BD.overlay_ball(img_left, [pbb])
        # color = BD.overlay_ball(img_left, [pby])

        # Find tool position, joint angles, and overlay
        if len(pbr) < 2:
            pass
        else:
            pt = BD.find_tool_position(pbr[0], pbr[1])    # tool position of pitch axis
            pt = np.array(pt) * 0.001  # (m)
            pt = BD.Rrc.dot(pt) + BD.trc
            q1, q2, q3 = dvrkKinematics.ik_position(pt)
            # print(q0*180/np.pi, q2*180/np.pi, q3)
            color = BD.overlay_tool_position(color, [q1,q2,q3], (0,255,0))

            # Find tool orientation, joint angles, and overlay
            if len(pbr) < 3:
                pass
            elif [pbr[2], pbg, pbb, pby].count([]) < 3:
                pass
            else:
                Rm = BD.find_tool_orientation(pbr[2], pbg, pbb, pby)    # orientation of the marker
                q4,q5,q6 = dvrkKinematics.ik_orientation(q1,q2,Rm)
                # print(q4*180/np.pi,q5*180/np.pi,q6*180/np.pi)
                # print(q5*180/np.pi)
                color = BD.overlay_tool(color, [q1, q2, q3, q4, q5, q6], (0,255,0))

        cv2.imshow("images", color)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            break