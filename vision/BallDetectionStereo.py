from FLSpegtransfer.path import *
import cv2
import numpy as np
from FLSpegtransfer.motion.dvrkKinematics import dvrkKinematics
from FLSpegtransfer.vision.AlliedVisionUtils import AlliedVisionUtils
import FLSpegtransfer.motion.dvrkVariables as dvrkVar
from sklearn import linear_model


class BallDetectionStereo:
    def __init__(self, Trc):
        # Transform
        # self.Tpc = Tpc  # from pegboard to camera
        self.Trc = Trc  # from camera to robot
        if Trc == []:
            pass
        else:
            self.Rrc = self.Trc[:3, :3]
            self.trc = self.Trc[:3, 3]

        # thresholding value
        self.__lower_red = np.array([0-20, 60, 50])
        self.__upper_red = np.array([0+20, 255, 255])
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

        # instances
        self.av = AlliedVisionUtils()

    def overlay_ball(self, img_color, pbs, which='left'):
        pbs = np.array(pbs)
        if pbs.size == 0:
            return img_color
        else:
            overlayed = img_color.copy()
            for i,pb in enumerate(pbs):
                # pb_img = self.world2pixel(pb[0], pb[1], pb[2], pb[3])
                pb_img = self.world2pixel_stereo(pb[0], pb[1], pb[2], pb[3], which)
                cv2.circle(overlayed, (pb_img[0], pb_img[1]), pb_img[2], (0, 255, 255), 2)
                cv2.circle(overlayed, (pb_img[0], pb_img[1]), 7, (0, 255, 255), -1)
                # cv2.putText(overlayed, str(i), (pb_img[0]+10, pb_img[1]), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 255, 255), 2, cv2.LINE_AA)
        return overlayed

    def overlay_dot(self, img_color, pnt_3D, text, which='left'):
        pnt = np.array(pnt_3D)
        if pnt.size == 0:
            return img_color
        else:
            overlayed = img_color.copy()
            img_pnt = self.av.world2pixel(pnt[0], pnt[1], pnt[2], 0, which)
            cv2.circle(overlayed, (img_pnt[0], img_pnt[1]), 7, (0, 255, 255), -1)
            cv2.putText(overlayed, text, (img_pnt[0] + 10, img_pnt[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        return overlayed

    def overlay_vector(self, img_color, pbs1, pbs2, which='left'):
        pbs1 = np.array(pbs1)
        pbs2 = np.array(pbs2)
        if pbs1.size == 0 or pbs2.size==0:
            return img_color
        else:
            overlayed = img_color.copy()
            for i, (pb1, pb2) in enumerate(zip(pbs1, pbs2)):
                # pb_img = self.world2pixel(pb[0], pb[1], pb[2], pb[3])
                pb_img1 = self.av.world2pixel(pb1[0], pb1[1], pb1[2], pb1[3], which)
                pb_img2 = self.av.world2pixel(pb2[0], pb2[1], pb2[2], pb2[3], which)
                # cv2.circle(overlayed, (pb_img[0], pb_img[1]), pb_img[2], (0, 255, 255), 2)
                # cv2.putText(overlayed, str(i), (pb_img[0]+10, pb_img[1]), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 255, 255), 2, cv2.LINE_AA)
                # cv2.circle(overlayed, (pb_img[0], pb_img[1]), 10, (0, 255, 255), -1)
                cv2.arrowedLine(overlayed, pb_img1[:2], pb_img2[:2], (0, 0, 255), 3, tipLength=0.05)
        return overlayed

    def overlay_tool(self, img_color, joint_angles, color):
        # 3D points w.r.t camera frame
        pb = self.Rrc.T.dot(np.array([0,0,0])-self.trc)*1000    # base position
        p5 = np.array(dvrkKinematics.fk(joint_angles,L1=dvrkVar.L1,L2=dvrkVar.L2,L3=0,L4=0))[:3, 3]
        p5 = self.Rrc.T.dot(np.array(p5)-self.trc)*1000  # pitch axis
        p6 = np.array(dvrkKinematics.fk(joint_angles,L1=dvrkVar.L1,L2=dvrkVar.L2,L3=dvrkVar.L3,L4=0))[:3, 3]
        p6 = self.Rrc.T.dot(np.array(p6)-self.trc)*1000  # yaw axis
        p7 = np.array(dvrkKinematics.fk(joint_angles,L1=dvrkVar.L1,L2=dvrkVar.L2,L3=dvrkVar.L3,L4=dvrkVar.L4+0.005))[:3, 3]
        p7 = self.Rrc.T.dot(np.array(p7)-self.trc)*1000  # tip

        pb_img = self.av.world2pixel(pb[0], pb[1], pb[2])
        p5_img = self.av.world2pixel(p5[0], p5[1], p5[2])
        p6_img = self.av.world2pixel(p6[0], p6[1], p6[2])
        p7_img = self.av.world2pixel(p7[0], p7[1], p7[2])

        overlayed = img_color.copy()
        self.drawline(overlayed, pb_img[0:2], p5_img[0:2], (0,255,0), 1, style='dotted', gap=8)
        cv2.circle(overlayed, p5_img[0:2], 2, color, 2)
        self.drawline(overlayed, p5_img[0:2], p6_img[0:2], (0,255,0), 1, style='dotted', gap=8)
        cv2.circle(overlayed, p6_img[0:2], 2, color, 2)
        self.drawline(overlayed, p6_img[0:2], p7_img[0:2], (0,255,0), 1, style='dotted', gap=8)
        cv2.circle(overlayed, p7_img[0:2], 2, color, 2)
        return overlayed

    def overlay_tool_position(self, img_color, joint_angles, color):
        q1, q2, q3 = joint_angles
        # 3D points w.r.t camera frame
        pb = self.Rrc.T.dot(np.array([0, 0, 0]) - self.trc) * 1000  # base position
        p5 = np.array(dvrkKinematics.fk([q1, q2, q3, 0, 0, 0], L1=dvrkVar.L1, L2=dvrkVar.L2, L3=0, L4=0))[:3, 3]
        p5 = self.Rrc.T.dot(np.array(p5) - self.trc) * 1000  # pitch axis

        pb_img = self.av.world2pixel(pb[0], pb[1], pb[2])
        p5_img = self.av.world2pixel(p5[0], p5[1], p5[2])

        overlayed = img_color.copy()
        self.drawline(overlayed, pb_img[0:2], p5_img[0:2], (0, 255, 0), 1, style='dotted', gap=8)
        cv2.circle(overlayed, p5_img[0:2], 2, color, 2)
        return overlayed

    def drawline(self, img, pt1, pt2, color, thickness=1, style='dotted', gap=20):
        dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
        pts = []
        for i in np.arange(0, dist, gap):
            r = i / dist
            x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
            y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
            pts.append((x, y))

        if style == 'dotted':
            for p in pts:
                cv2.circle(img, p, thickness, color, -1)
        elif style == 'dashed':
            st = pts[0]
            ed = pts[0]
            i = 0
            for p in pts:
                st = ed
                ed = p
                if i % 2 == 1:
                    cv2.line(img, st, ed, color, thickness)
                i += 1

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
        masked = cv2.morphologyEx(masked, cv2.MORPH_OPEN, kernel, iterations=3)
        return masked

    def find_balls2D(self, img_color, color, visualize=False):
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
        pb = []
        for c in cnts:
            if len(c) < 100:
            # if cv2.contourArea(c) > 500 # thresholding by area is more accurate
                pass
            else:
                # if visualize:
                img_color_copy = np.copy(img_color)
                cv2.drawContours(img_color_copy, [c], -1, (0, 255, 255), 1)
                if visualize:
                    cv2.imshow("", img_color_copy)
                    cv2.waitKey(0)

                # Get the pixel coordinates
                args = np.squeeze(c)
                x = args[:, 0].reshape(-1,1)
                y = args[:, 1].reshape(-1,1)

                # Linear regression to fit the circle into the image points
                X_ = np.concatenate((x, y, np.ones_like(x)), axis=1)
                y_ = x**2 + y**2
                ransac = linear_model.RANSACRegressor(min_samples=3, residual_threshold=300)
                ransac.fit(X_, y_)
                c0, c1, _ = ransac.estimator_.coef_[0]
                c2 = ransac.estimator_.intercept_[0]
                xc = c0/2
                yc = c1/2
                rc = np.sqrt(c2 + xc**2 + yc**2)
                inlier_mask = ransac.inlier_mask_
                nb_inlier = len(inlier_mask[inlier_mask])
                if visualize:
                    img_color_copy = np.copy(img_color)
                    cv2.circle(img_color_copy, (int(xc), int(yc)), int(rc), (0, 255, 255), 2)
                    cv2.circle(img_color_copy, (int(xc), int(yc)), 3, (0, 255, 255), -1)
                    cv2.imshow("", img_color_copy)
                    cv2.waitKey(0)
                if (nb_inlier > 50) and (20 < rc < 150):
                    pb.append([xc, yc, rc])
        if len(pb) >= 2:
            # sort by radius
            pb = np.array(pb)
            arg = np.argsort(pb[:, 2])[::-1]
            return pb[arg]
        else:
            return pb

    def find_balls3D(self, img_left, img_right, color, visualize=False):
        pbl = np.array(self.find_balls2D(img_left, color, visualize))
        pbr = np.array(self.find_balls2D(img_right, color, visualize))
        pbs = []
        for pl, pr in zip(pbl, pbr):
            Xc, Yc, Zc, Rc = self.av.pixel2world(pl, pr)
            pbs.append([Xc, Yc, Zc, Rc])
        return pbs

    # Get tool position of the pitch axis from two ball positions w.r.t. camera base coordinate
    def find_tool_pitch(self, pb1, pb2):
        pb1 = np.asarray(pb1[0:3], dtype=float)
        pb2 = np.asarray(pb2[0:3], dtype=float)
        p_pitch = ((self.Lbb+self.Lbp)*pb2-self.Lbp*pb1)/self.Lbb
        return p_pitch    # (m), w.r.t. camera base coordinate