from FLSpegtransfer.vision.AlliedVisionUtils import AlliedVisionUtils
import cv2
import numpy as np
from skimage.morphology import skeletonize
import FLSpegtransfer.utils.CmnUtil as U
from FLSpegtransfer.utils.ImgUtils import ImgUtils
from scipy.spatial import distance
import time

class NeedleTrackingStereoNumerical:
    def __init__(self):
        # thresholding value
        self.__lower_red = np.array([0 - 20, 100, 100])
        self.__upper_red = np.array([0 + 20, 255, 255])
        self.__lower_green = np.array([60 - 40, 100, 50])
        self.__upper_green = np.array([60 - 10, 255, 255])
        self.__lower_blue = np.array([120 - 20, 70, 30])
        self.__upper_blue = np.array([120 + 20, 255, 255])
        self.__lower_yellow = np.array([30 - 10, 130, 60])
        self.__upper_yellow = np.array([30 + 10, 255, 255])

        # instances
        self.av = AlliedVisionUtils()

        # hyperparameter
        self.dt = np.array([1, 1, 1, 1000, 5000, 5000]) * 10 ** (-9)  # step size (gain)
        self.pose = [0.0, 0.0, 0.3, 0.0, 0.0, 0.0]  # initial guess of needle pose

        # data members
        self.p_needle = self.sample_3Dpoints(nb_pnts=100, radius=0.020, st_angle=180+20, ed_angle=360-20)

    def mask_color(self, img_color, color, visualize):
        # define hsv_range
        if color == 'red':
            hsv_range = [self.__lower_red, self.__upper_red]
        elif color == 'green':
            hsv_range = [self.__lower_green, self.__upper_green]
        elif color == 'blue':
            hsv_range = [self.__lower_blue, self.__upper_blue]
        elif color == 'yellow':
            hsv_range = [self.__lower_yellow, self.__upper_yellow]
        else:
            hsv_range = []

        # 2D color masking
        img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
        masked = cv2.inRange(img_hsv, hsv_range[0], hsv_range[1])

        # filtering
        # masked = cv2.erode(masked, kernel, iterations=1)
        # masked = cv2.dilate(masked, kernel, iterations=1)
        # masked = cv2.morphologyEx(masked, cv2.MORPH_OPEN, kernel, iterations=3)
        if visualize:
            cv2.imshow("", masked)
            cv2.waitKey(0)
        return masked

    def mask_needle(self, img_color, img_masked, visualize):
        # Find contours
        cnts, _ = cv2.findContours(img_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        cnts_needle = [c for c in cnts if cv2.contourArea(c) > 80]  # threshold by contour size
        if visualize:
            img_color_copy = img_color.copy()
            cv2.drawContours(img_color_copy, cnts_needle, -1, (0, 255, 255), 1)
            cv2.imshow("", img_color_copy)
            cv2.waitKey(0)

        # Clean the masked image
        infilled = np.zeros(np.shape(img_color), np.uint8)
        cv2.drawContours(infilled, cnts_needle, -1, (255, 255, 255), -1)
        infilled = cv2.cvtColor(infilled, cv2.COLOR_BGR2GRAY)
        needle_masked = cv2.bitwise_and(img_masked, img_masked, mask=infilled)
        if visualize:
            cv2.imshow("", needle_masked)
            cv2.waitKey(0)
        cnts_needle = np.concatenate(cnts_needle, axis=0)
        return needle_masked, cnts_needle   # masked image, contours

    def skeletonize(self, img_masked, visualize):
        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(img_masked, cv2.MORPH_CLOSE, kernel, iterations=5)
        closed = closed > 200
        skeleton = skeletonize(closed)
        skeleton = skeleton.astype(np.uint8)  # convert to an unsigned byte
        skeleton *= 255
        # needle_skeleton = cv2.ximgproc.thinning(needle_masked_closed)     # opencv method is slow (0.1~0.2sec)
        if visualize:
            cv2.imshow("", skeleton)
            cv2.waitKey(0)
        return skeleton

    def get_coordinate(self, img_binary):
        # Get the pixel coordinates of needle skeleton
        args = np.argwhere(img_binary == 255)
        x = args[:, 1].reshape(1, -1)
        y = args[:, 0].reshape(1, -1)
        pnts = np.concatenate((x, y), axis=0)
        return pnts

    # sample 2D image points for clean needle img pnts
    def sample_2Dpoints(self, cx, cy, a, b, nb_pnts):
        theta = np.linspace(0, 2 * np.pi, num=nb_pnts)
        x = (a * np.cos(theta) + cx).reshape(1, -1)
        y = (b * np.sin(theta) + cy).reshape(1, -1)
        p = np.concatenate((x, y), axis=0)
        return p  # dim(p) = 2xn

    # sample 3D image points in ellipse w.r.t needle frame
    def sample_3Dpoints(self, nb_pnts, radius, st_angle, ed_angle):
        theta = np.linspace(np.deg2rad(st_angle), np.deg2rad(ed_angle), num=nb_pnts)
        x = (radius * np.cos(theta)).reshape(1, -1)
        y = (radius * np.sin(theta)).reshape(1, -1)
        z = np.zeros_like(x)
        p = np.concatenate((x, y, z), axis=0)
        return p  # dim(p) = 3xn

    # rotate 2D image points
    def rotate_points(self, cx, cy, pnts, angle):   # angle in (rad)
        centered = pnts - np.array([[cx],[cy]])
        Rz = U.Rz(angle)[:2,:2]    # coordinate transform (same with point transform?)
        rotated = Rz.dot(centered)
        return rotated + np.array([[cx],[cy]])

    # get clean points
    def clean_pnts(self, img_nd_masked, img_nd_skeleton):
        pnts_skeleton = self.get_coordinate(img_nd_skeleton)
        (cx, cy), (a, b), angle = cv2.fitEllipse(pnts_skeleton.T)  # Fit ellipse (OpenCV method)
        a = a / 2
        b = b / 2
        pnts_sample = self.sample_2Dpoints(cx, cy, a, b, nb_pnts=200)
        pnts_rotated = self.rotate_points(cx, cy, pnts_sample, np.deg2rad(angle))
        img_nd_skeleton_cleaned = cv2.bitwise_and(img_nd_masked, self.overlay_pnts(np.zeros_like(img_nd_masked), pnts_rotated, pnts_rotated))
        pnts_cleaned = self.get_coordinate(img_nd_skeleton_cleaned)
        return pnts_cleaned, (cx, cy, a, b, angle)

    def find_needle2D(self, img_color, color, visualize=False):
        if visualize:
            cv2.imshow("", img_color)
            cv2.waitKey(0)

        # color masking
        img_masked = self.mask_color(img_color, color, visualize)
        needle_masked, cnts_needle = self.mask_needle(img_color, img_masked, visualize)
        needle_skeleton = self.skeletonize(needle_masked, visualize)
        if visualize:
            overlay_skeleton = cv2.add(img_color, cv2.cvtColor(needle_skeleton, cv2.COLOR_GRAY2BGR))
            cv2.imshow("", overlay_skeleton)
            cv2.waitKey(0)
        return needle_masked, needle_skeleton

    # transform 3D points w.r.t needle frame into left/right image frame
    def pose_transform(self, nP, pose_needle):
        tx, ty, tz, alpha, beta, gamma = pose_needle
        Rz = U.Rz(alpha)
        Ry = U.Ry(beta)
        Rx = U.Rx(gamma)
        cRn = Rz.dot(Ry).dot(Rx)
        ctn = np.array([tx,ty,tz]).reshape(1,-1)
        cTn = np.block([[cRn, ctn.T], [0, 0, 0, 1]])    # transform from camera to needle
        cP = cRn.dot(nP) + ctn.T    # 3D points w.r.t camera frame

        # 2D image points projected on left/right image frame
        pnts_L, _ = self.av.world2pixel(cP, which='left')
        pnts_R, _ = self.av.world2pixel(cP, which='right')
        return [pnts_L, pnts_R], cTn

    # calculate error between image points
    def error(self, pnts_des, pose_needle):
        pnts_act, cTn = self.pose_transform(self.p_needle, pose_needle)
        pnts_des_L, pnts_des_R = pnts_des
        pnts_act_L, pnts_act_R = pnts_act
        errors_L = distance.cdist(pnts_act_L.T, pnts_des_L.T).min(axis=1).sum()
        errors_R = distance.cdist(pnts_act_R.T, pnts_des_R.T).min(axis=1).sum()
        return errors_L+errors_R

    def error_grad(self, pnts_des, pose_needle):
        tx, ty, tz, alpha, beta, gamma = pose_needle
        # variation size
        dtx = 0.001 # (m)
        dty = 0.001
        dtz = 0.001
        da = np.deg2rad(5)   # (rad)
        db = np.deg2rad(10)
        dg = np.deg2rad(10)
        derr_tx = (self.error(pnts_des, [tx+dtx, ty, tz, alpha, beta, gamma]) - self.error(pnts_des, [tx-dtx, ty, tz, alpha, beta, gamma]))/(2*dtx)
        derr_ty = (self.error(pnts_des, [tx, ty+dty, tz, alpha, beta, gamma]) - self.error(pnts_des, [tx, ty-dty, tz, alpha, beta, gamma]))/(2*dty)
        derr_tz = (self.error(pnts_des, [tx, ty, tz+dtz, alpha, beta, gamma]) - self.error(pnts_des, [tx, ty, tz-dtz, alpha, beta, gamma]))/(2*dtz)
        derr_a = (self.error(pnts_des, [tx, ty, tz, alpha+da, beta, gamma]) - self.error(pnts_des, [tx, ty, tz, alpha-da, beta, gamma]))/(2*da)
        derr_b = (self.error(pnts_des, [tx, ty, tz, alpha, beta+db, gamma]) - self.error(pnts_des, [tx, ty, tz, alpha, beta-db, gamma]))/(2*db)
        derr_g = (self.error(pnts_des, [tx, ty, tz, alpha, beta, gamma+dg]) - self.error(pnts_des, [tx, ty, tz, alpha, beta, gamma-dg]))/(2*dg)
        return np.array([derr_tx, derr_ty, derr_tz, derr_a, derr_b, derr_g])

    def find_needle3D(self, img_left, img_right, color, visualize=False):
        img_nd_masked_L, img_nd_skeleton_L = self.find_needle2D(img_left, color, visualize=False)
        img_nd_masked_R, img_nd_skeleton_R = self.find_needle2D(img_right, color, visualize=False)
        # if visualize:
        #     stacked = ImgUtils.stack_stereo_img(nd_skeleton_L, nd_skeleton_R, scale=0.7)
        #     cv2.imshow("", stacked)
        #     cv2.waitKey(0)

        # clean image points
        pnts_des_L, ellipse_L = self.clean_pnts(img_nd_masked_L, img_nd_skeleton_L)
        pnts_des_R, ellipse_R = self.clean_pnts(img_nd_masked_R, img_nd_skeleton_R)

        st = time.time()
        while True:
            pnts_des = (pnts_des_L, pnts_des_R)
            self.pose = self.pose - self.error_grad(pnts_des, self.pose)*self.dt
            error = self.error(pnts_des, self.pose)
            if error < 500:
                break
            print (self.pose, error)
            if visualize:
                pnts_act, cTn = self.pose_transform(self.p_needle, self.pose)
                pnts_act_L, pnts_act_R = pnts_act
                img_left_copy = img_left.copy()
                img_right_copy = img_right.copy()
                img_left_copy = self.overlay_pnts(img_left_copy, pnts_des_L, pnts_act_L)
                img_right_copy = self.overlay_pnts(img_right_copy, pnts_des_R, pnts_act_R)
                # img_left = self.draw_axes(img_left, p_org, [nx, ny, nz], length=10, thickness=2)
                stacked = ImgUtils.stack_stereo_img(img_left_copy, img_right_copy, scale=0.7)
                cv2.imshow("", stacked)
                cv2.waitKey(5)
        print (time.time() - st)

    def overlay_pnts(self, image, pnts_des, pnts_act):
        image_copy = image.copy()
        pnts_des = pnts_des.astype(int)
        pnts_act = pnts_act.astype(int)
        if len(np.shape(image)) == 3:
            for pd, pa in zip(pnts_des.T, pnts_act.T):
                cv2.circle(image_copy, (pd[0], pd[1]), 2, (0, 255, 0), -1)  # green
                cv2.circle(image_copy, (pa[0], pa[1]), 2, (0, 0, 255), -1)  # red
            # image_copy[pnts[1, :], pnts[0, :]] = [0, 255, 0]
        elif len(np.shape(image)) == 2:
            image_copy[pnts_des[1, :], pnts_des[0, :]] = 255
            image_copy[pnts_act[1, :], pnts_act[0, :]] = 255
        else:
            raise ValueError
        return image_copy

    def draw_axes(self, image, p_org, normals, length, thickness):
        nx, ny, nz = normals
        pnt_org, _ = self.av.world2pixel(p_org, which='left')
        pnt_nx, _ = self.av.world2pixel(p_org + nx / 1000 * length, which='left')
        pnt_ny, _ = self.av.world2pixel(p_org + ny / 1000 * length, which='left')
        pnt_nz, _ = self.av.world2pixel(p_org + nz / 1000 * length, which='left')

        pnt_org = pnt_org.astype(int).squeeze()
        pnt_nx = pnt_nx.astype(int).squeeze()
        pnt_ny = pnt_ny.astype(int).squeeze()
        pnt_nz = pnt_nz.astype(int).squeeze()

        img = image.copy()
        cv2.arrowedLine(img, tuple(pnt_org), tuple(pnt_nx), (0, 0, 255), thickness)
        cv2.arrowedLine(img, tuple(pnt_org), tuple(pnt_ny), (0, 255, 0), thickness)
        cv2.arrowedLine(img, tuple(pnt_org), tuple(pnt_nz), (255, 0, 0), thickness)
        return img