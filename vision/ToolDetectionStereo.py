from FLSpegtransfer.path import *
from FLSpegtransfer.motion.dvrkKinematics import dvrkKinematics
import FLSpegtransfer.motion.dvrkVariables as dvrkVar
import numpy as np
import cv2


class ToolDetection:
    def __init__(self, Trc):
        # Transform from camera to robot
        self.Trc = Trc
        if Trc == []:
            pass
        else:
            self.Rrc = self.Trc[:3, :3]
            self.trc = self.Trc[:3, 3]

        # camera matrices
        path = 'calibration_files/alliedvision/'
        self.K = np.load(root+path+'K.npy')
        self.R = np.load(root+path+'R.npy')
        self.T = np.load(root+path+'T.npy')
        self.Q = np.load(root+path+'Q.npy')
        self.P1 = np.load(root+path+'P1.npy')
        self.P2 = np.load(root+path+'P2.npy')
        self.fx = self.K[:, 0, 0]
        self.fy = self.K[:, 1, 1]
        self.cx = self.K[:, 0, 2]
        self.cy = self.K[:, 1, 2]

        # thresholding value
        self.__lower_red = np.array([0-10, 80, 80])
        self.__upper_red = np.array([0+10, 255, 255])
        self.__lower_green = np.array([60-40, 100, 50])
        self.__upper_green = np.array([60-10, 255, 255])
        self.__lower_blue = np.array([120-20, 130, 40])
        self.__upper_blue = np.array([120+20, 255, 255])
        self.__lower_yellow = np.array([30-10, 130, 60])
        self.__upper_yellow = np.array([30+10, 255, 255])
        # radius of sphere fiducials = [12.0, 10.0, 8.0, 8.0, 8.0, 8.0]    # (mm)

        # dimension of tool
        self.d = 35       # length of coordinate (mm)
        self.Lbb = 0.050  # ball1 ~ ball2 (m)
        self.Lbp = 0.017  # ball2 ~ pitch (m)
        # self.Lbp = 0.017 + dvrkVar.L3 + dvrkVar.L4

        # crop parameters
        self.xcr = [0, 0]
        self.wcr = [0, 0]
        self.ycr = [0, 0]
        self.hcr = [0, 0]

    def pixel2world_stereo(self, pl, pr):
        pixel_homo = np.array([pl[0], pl[1], pl[0] - pr[0], 1]).T   # [x, y, disparity, 1].T
        P = self.Q.dot(pixel_homo)
        X = P[0] / P[3]
        Y = P[1] / P[3]
        Z = P[2] / P[3]
        f = self.Q[2,3]
        R = (pl[2]+pr[2])/2 * Z/f
        return X, Y, Z, R

    def world2pixel_stereo(self, X, Y, Z, R=0, which='left'):
        P = np.array([X, Y, Z, 1]).T
        if which=='left':
            p = self.P1.dot(P)
            fx = self.P1[0, 0]
            fy = self.P1[1, 1]
        elif which=='right':
            p = self.P2.dot(P)
            fx = self.P2[0, 0]
            fy = self.P2[1, 1]
        x = p[0] / p[2]
        y = p[1] / p[2]
        r = (fx + fy)/2 * R / Z
        return int(x), int(y), int(r)

    def overlay_tooltip(self, img_color, pt, which='left'):
        pt = np.array(pt)
        if pt.size == 0:
            return img_color
        else:
            overlayed = img_color.copy()
            pb_img = self.world2pixel_stereo(pt[0], pt[1], pt[2], 0, which)
            cv2.circle(overlayed, (pb_img[0], pb_img[1]), 7, (0, 255, 255), -1)
        return overlayed

    def overlay_vector(self, img_color, pbs1, pbs2, which='left'):
        pbs1 = np.array(pbs1)
        pbs2 = np.array(pbs2)
        if pbs1.size == 0 or pbs2.size==0:
            return img_color
        else:
            overlayed = img_color.copy()
            for i, (pb1, pb2) in enumerate(zip(pbs1, pbs2)):
                pb_img1 = self.world2pixel_stereo(pb1[0], pb1[1], pb1[2], pb1[3], which)
                pb_img2 = self.world2pixel_stereo(pb2[0], pb2[1], pb2[2], pb2[3], which)
                cv2.arrowedLine(overlayed, pb_img1[:2], pb_img2[:2], (0, 255, 0), 5)
        return overlayed

    # def overlay_tool(self, img_color, joint_angles, color):
    #     # 3D points w.r.t camera frame
    #     pb = self.Rrc.T.dot(np.array([0,0,0])-self.trc)*1000    # base position
    #     p5 = dvrkKinematics.fk_position(joint_angles,L1=dvrkVar.L1,L2=dvrkVar.L2,L3=0,L4=0)
    #     p5 = self.Rrc.T.dot(np.array(p5)-self.trc)*1000  # pitch axis
    #     p6 = dvrkKinematics.fk_position(joint_angles,L1=dvrkVar.L1,L2=dvrkVar.L2,L3=dvrkVar.L3,L4=0)
    #     p6 = self.Rrc.T.dot(np.array(p6)-self.trc)*1000  # yaw axis
    #     p7 = dvrkKinematics.fk_position(joint_angles,L1=dvrkVar.L1,L2=dvrkVar.L2,L3=dvrkVar.L3,L4=dvrkVar.L4+0.005)
    #     p7 = self.Rrc.T.dot(np.array(p7)-self.trc)*1000  # tip
    #
    #     pb_img = self.world2pixel(pb[0], pb[1], pb[2])
    #     p5_img = self.world2pixel(p5[0], p5[1], p5[2])
    #     p6_img = self.world2pixel(p6[0], p6[1], p6[2])
    #     p7_img = self.world2pixel(p7[0], p7[1], p7[2])
    #
    #     overlayed = img_color.copy()
    #     self.drawline(overlayed, pb_img[0:2], p5_img[0:2], (0,255,0), 1, style='dotted', gap=8)
    #     cv2.circle(overlayed, p5_img[0:2], 2, color, 2)
    #     self.drawline(overlayed, p5_img[0:2], p6_img[0:2], (0,255,0), 1, style='dotted', gap=8)
    #     cv2.circle(overlayed, p6_img[0:2], 2, color, 2)
    #     self.drawline(overlayed, p6_img[0:2], p7_img[0:2], (0,255,0), 1, style='dotted', gap=8)
    #     cv2.circle(overlayed, p7_img[0:2], 2, color, 2)
    #     return overlayed
    #
    # def overlay_tool_position(self, img_color, joint_angles, color):
    #     q1, q2, q3 = joint_angles
    #     # 3D points w.r.t camera frame
    #     pb = self.Rrc.T.dot(np.array([0, 0, 0]) - self.trc) * 1000  # base position
    #     p5 = dvrkKinematics.fk_position([q1, q2, q3, 0, 0, 0], L1=dvrkVar.L1, L2=dvrkVar.L2, L3=0, L4=0)
    #     p5 = self.Rrc.T.dot(np.array(p5) - self.trc) * 1000  # pitch axis
    #
    #     pb_img = self.world2pixel(pb[0], pb[1], pb[2])
    #     p5_img = self.world2pixel(p5[0], p5[1], p5[2])
    #
    #     overlayed = img_color.copy()
    #     self.drawline(overlayed, pb_img[0:2], p5_img[0:2], (0, 255, 0), 1, style='dotted', gap=8)
    #     cv2.circle(overlayed, p5_img[0:2], 2, color, 2)
    #     return overlayed
    #
    # def drawline(self, img, pt1, pt2, color, thickness=1, style='dotted', gap=20):
    #     dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
    #     pts = []
    #     for i in np.arange(0, dist, gap):
    #         r = i / dist
    #         x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
    #         y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
    #         pts.append((x, y))
    #
    #     if style == 'dotted':
    #         for p in pts:
    #             cv2.circle(img, p, thickness, color, -1)
    #     elif style == 'dashed':
    #         st = pts[0]
    #         ed = pts[0]
    #         i = 0
    #         for p in pts:
    #             st = ed
    #             ed = p
    #             if i % 2 == 1:
    #                 cv2.line(img, st, ed, color, thickness)
    #             i += 1

    @classmethod
    def transform(cls, points, T):
        R = T[:3, :3]
        t = T[:3, -1]
        return R.dot(points.T).T + t.T

    def mask_image(self, img_color, color):
        # define hsv_range
        if color == 'red':      hsv_range = [self.__lower_red, self.__upper_red]
        elif color == 'green':  hsv_range = [self.__lower_green, self.__upper_green]
        elif color == 'blue':   hsv_range = [self.__lower_blue, self.__upper_blue]
        elif color == 'yellow': hsv_range = [self.__lower_yellow, self.__upper_yellow]
        else:   hsv_range = []

        # 2D color masking
        img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
        masked = cv2.inRange(img_hsv, hsv_range[0], hsv_range[1])

        # filtering
        kernel = np.ones((3, 3), np.uint8)
        # masked = cv2.erode(masked, kernel, iterations=2)
        # masked = cv2.dilate(masked, kernel, iterations=1)
        masked = cv2.morphologyEx(masked, cv2.MORPH_OPEN, kernel, iterations=1)
        return masked

    def find_tool2D(self, img_color, color, visualize=False):
        if visualize:
            cv2.imshow("", img_color)
            cv2.waitKey(0)

        # color masking
        masked = self.mask_image(img_color, color)
        if visualize:
            cv2.imshow("", masked)
            cv2.waitKey(0)

        # Find contours
        cnts, _ = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        if len(cnts) ==0:
            return []
        else:
            # if visualize:
            img_color_copy = np.copy(img_color)
            cv2.drawContours(img_color_copy, [cnts[0]], -1, (0, 255, 255), 1)
            if visualize:
                cv2.imshow("", img_color_copy)
                cv2.waitKey(0)

            # Get the pixel coordinates
            args = np.squeeze(cnts[0])
            x = args[:, 0].reshape(-1,1)
            y = args[:, 1].reshape(-1,1)
            k = y - x
            argmax = np.argmax(k)
            pt = args[argmax]

            if visualize:
                img_color_copy = np.copy(img_color)
                cv2.circle(img_color_copy, (int(pt[0]), int(pt[1])), 5, (0, 255, 255), -1)
                cv2.imshow("", img_color_copy)
                cv2.waitKey(0)
            return pt

    def find_tool3D(self, img_left, img_right, color, visualize=False):
        pl = np.array(self.find_tool2D(img_left, color, visualize))
        pr = np.array(self.find_tool2D(img_right, color, visualize))
        if len(pl) == 0 or len(pr) == 0:
            return []
        else:
            return self.pixel2world_stereo([pl[0], pl[1], 0.0], [pr[0], pr[1], 0.0])

    def pose_estimation(pbr, pbg, pbb, pby, use_Trc):  # Find tool position, joint angles
        pt = []
        q_phy = []
        if len(pbr) < 2:
            pass
        else:
            pt = bd.find_tool_pitch(pbr[0], pbr[1])  # tool position of pitch axis
            pt = np.array(pt) * 0.001  # (m)
            if use_Trc:
                pt = bd.Rrc.dot(pt) + bd.trc
                qp1, qp2, qp3 = dvrkKinematics.ik_position_straight(pt, L3=0, L4=0)  # position of pitch axis

                # Find tool orientation, joint angles, and overlay
                temp = [pbr[2], pbg, pbb, pby]
                if len(pbr) < 3:
                    qp4 = 0.0;
                    qp5 = 0.0;
                    qp6 = 0.0
                elif temp.count([]) > 2:
                    qp4 = 0.0;
                    qp5 = 0.0;
                    qp6 = 0.0
                else:
                    Rm = bd.find_tool_orientation(pbr[2], pbg, pbb, pby)  # orientation of the marker
                    qp4, qp5, qp6 = dvrkKinematics.ik_orientation(qp1, qp2, Rm)
                q_phy = [qp1, qp2, qp3, qp4, qp5, qp6]
            else:
                q_phy = []
        return pt, q_phy