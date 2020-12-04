from FLSpegtransfer.vision.AlliedVisionUtils import AlliedVisionUtils
import FLSpegtransfer.utils.CmnUtil as U
from FLSpegtransfer.utils.ImgUtils import ImgUtils
from FLSpegtransfer.vision.PCLRegistration import PCLRegistration
from FLSpegtransfer.utils.plot import *
from FLSpegtransfer.utils.Filter.Kalman import Kalman
from skimage.morphology import skeletonize
import open3d.cpu.pybind.pipelines.registration as o3d_reg
import open3d as o3d
from pycpd import AffineRegistration
from probreg import l2dist_regs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import copy
import cv2
import time, threading

class NeedleTrackingStereo:
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

        # data members
        self.nb_pnts = 50
        self.T = np.identity(4)

        # fig = plt.figure()
        # self.ax = fig.add_subplot(111, projection='3d')
        # plt.ion()
        # plt.show()

        # instances
        self.av = AlliedVisionUtils()
        self.pnts_model = self.sample_3Dpoints(nb_pnts=self.nb_pnts, radius=.017, st_angle=np.deg2rad(0+10), ed_angle=np.deg2rad(180-10))
        A = np.eye(3)
        H = np.eye(3)
        Q = np.diag([1.0, 1.0, 1.0])
        R = np.diag([20.0, 20.0, 20.0])
        x0 = [1.0, 0.0, 0.0]
        P0 = np.eye(3)*100
        self.Kalman = Kalman(A, H, Q, R, x0, P0)

    # sample 2D image points for clean needle img pnts
    @classmethod
    def sample_2Dpoints(cls, cx, cy, a, b, nb_pnts):
        theta = np.linspace(0, 2 * np.pi, num=nb_pnts)
        x = (a * np.cos(theta) + cx).reshape(1, -1)
        y = (b * np.sin(theta) + cy).reshape(1, -1)
        p = np.concatenate((x, y), axis=0)
        return p  # dim(p) = 2xn

    # sample 3D image points in ellipse w.r.t needle frame
    @classmethod
    def sample_3Dpoints(cls, nb_pnts, radius, st_angle, ed_angle):
        theta = np.linspace(st_angle, ed_angle, num=nb_pnts)
        x = (radius * np.cos(theta)).reshape(1, -1)
        y = (radius * np.sin(theta)).reshape(1, -1)
        z = np.ones_like(x)*0.2    # 50 cm far from lens
        p = np.concatenate((x, y, z), axis=0)
        return p  # dim(p) = 3xn

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

        # Clean the masked image
        infilled = np.zeros(np.shape(img_color), np.uint8)
        cv2.drawContours(infilled, cnts_needle, -1, (255, 255, 255), -1)
        needle_masked = cv2.cvtColor(infilled, cv2.COLOR_BGR2GRAY)

        # needle_masked = cv2.bitwise_and(img_masked, img_masked, mask=infilled)
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

    # rotate 2D image points
    def rotate_points(self, cx, cy, pnts, angle):   # angle in (rad)
        centered = pnts - np.array([[cx],[cy]])
        Rz = U.Rz(angle)[:2,:2]    # coordinate transform (same with point transform?)
        rotated = Rz.dot(centered)
        return rotated + np.array([[cx],[cy]])

    # get clean points
    def clean_pnts(self, img_nd_masked):
        pnts_skeleton = self.get_coordinate(img_nd_masked)
        (cx, cy), (a, b), angle = cv2.fitEllipse(pnts_skeleton.T)  # Fit ellipse (OpenCV method)
        a = a / 2
        b = b / 2
        pnts_sample = self.sample_2Dpoints(cx, cy, a, b, nb_pnts=self.nb_pnts)
        pnts_rotated = self.rotate_points(cx, cy, pnts_sample, np.deg2rad(angle))
        img_nd_skeleton_cleaned = cv2.bitwise_and(img_nd_masked, self.overlay_pnts(np.zeros_like(img_nd_masked), pnts_rotated))
        pnts_cleaned = self.get_coordinate(img_nd_skeleton_cleaned)
        return pnts_cleaned, (cx, cy, a, b, angle)

    @classmethod
    def registration(cls, source, target, T0=[], use_svr=False, save_image=False, visualize=False):
        source, _ = PCLRegistration.convert(source)
        target, _ = PCLRegistration.convert(target)
        src = copy.deepcopy(source)
        tgt = copy.deepcopy(target)
        # T = np.identity(4)

        if np.allclose(T0, np.identity(4)):
            print ("init")
            # coarse registration by mean matching
            pnt_src = np.asarray(src.points)
            pnt_tgt = np.asarray(tgt.points)
            mean_source = pnt_src.mean(axis=0)
            mean_target = pnt_tgt.mean(axis=0)
            t = mean_target - mean_source
            T_temp = np.identity(4)
            T_temp[:3, -1] = t
            src.transform(T_temp)
            T0 = copy.deepcopy(T_temp).dot(T0)

        if use_svr:
            # svr registration
            tf_param = l2dist_regs.registration_svr(src, tgt)
            T_temp[:3, :3] = tf_param.rot
            T_temp[:3, -1] = tf_param.t
            src.transform(T_temp)
            T = copy.deepcopy(T_temp).dot(T)
        else:
            # Point-to-point ICP
            # threshold = [20, 10, 5, 2, 1, 0.5, 0.2]
            # for i in range(len(threshold)):
            threshold = 0.1
            reg_p2p = o3d_reg.registration_icp(src, tgt, threshold, T0, o3d_reg.TransformationEstimationPointToPoint())

            # src.transform(reg_p2p.transformation)
            # T = copy.deepcopy(reg_p2p.transformation).dot(T)
            T = copy.deepcopy(reg_p2p.transformation)
        return T

    def transform_pnts(self, T, pnts):
        return T[:3, :3].dot(pnts).T + T[:3, -1].T

    def find_needle2D(self, img_color, color, visualize=False):
        if visualize:
            cv2.imshow("", img_color)
            cv2.waitKey(0)

        # color masking
        img_masked = self.mask_color(img_color, color, visualize)
        needle_masked, cnts_needle = self.mask_needle(img_color, img_masked, visualize)
        needle_skeleton = []
        return needle_masked, needle_skeleton

    def find_needle3D(self, img_left, img_right, color, visualize=False):
        img_nd_masked_L, img_nd_skeleton_L = self.find_needle2D(img_left, color, visualize=False)
        img_nd_masked_R, img_nd_skeleton_R = self.find_needle2D(img_right, color, visualize=False)

        # visualize masked images
        if visualize:
            stacked = ImgUtils.stack_stereo_img(img_nd_masked_L, img_nd_masked_R, scale=0.7)
            cv2.imshow("", stacked)
            cv2.waitKey(0)

        # Clean image points
        pnts_cln_L, ellipse_L = self.clean_pnts(img_nd_masked_L)
        pnts_cln_R, ellipse_R = self.clean_pnts(img_nd_masked_R)
        pnts_cln_R[1,:] -= 3  # for adjusting y-offset in stereo calibration

        # Get affine transform using CPD
        reg = AffineRegistration(**{'X': pnts_cln_R.T, 'Y': pnts_cln_L.T})
        TY, (B, t) = reg.register()  # affine transform matrix A = B.T
        A = B.T
        A[1,0] = 0.0
        A[1,1] = 1.0
        t[1] = 0

        # Kalman Filtering
        sx = A[0,0]
        hx = A[0,1]
        tx = t[0]
        z = [sx, hx, tx]
        filtered, _ = self.Kalman.estimate(z)
        A[0,0] = filtered[0]
        A[0,1] = filtered[1]
        t[0] = filtered[2]
        pnts_cln_R_new = A.dot(pnts_cln_L) + t.reshape(2, 1)

        # Visualize affine transform
        if visualize:
            img_left_copy = self.overlay_pnts(img_left.copy(), pnts_cln_L, color=(0, 255, 0))   # green
            img_right_copy = self.overlay_pnts(img_right.copy(), pnts_cln_R, color=(0, 255, 0))   # green
            img_right_copy = self.overlay_pnts(img_right_copy, pnts_cln_R_new, color=(0, 0, 255))  # red
            stacked = ImgUtils.stack_stereo_img(img_left_copy, img_right_copy, scale=0.7)
            cv2.imshow("", stacked)
            cv2.waitKey(0)

        # Reconstruct 3D points
        pnts_3D = self.av.pixel2world(pnts_cln_L, pnts_cln_R_new)

        # ICP Registration of needle model to 3D points
        pcl_source = o3d.geometry.PointCloud()
        pcl_target = o3d.geometry.PointCloud()
        pcl_source.points = o3d.utility.Vector3dVector(self.pnts_model.T)
        pcl_target.points = o3d.utility.Vector3dVector(pnts_3D.T)
        self.T = self.registration(pcl_source, pcl_target, T0=self.T, use_svr=False, save_image=False, visualize=False)

        # Visualize the projection of the 3D needle model
        pnts_model_tr = self.transform_pnts(self.T, self.pnts_model).T
        pnts_model_L, _ = self.av.world2pixel(pnts_model_tr, which='left')
        pnts_model_R, _ = self.av.world2pixel(pnts_model_tr, which='right')
        img_left = self.overlay_pnts(img_left, pnts_model_L)
        img_right = self.overlay_pnts(img_right, pnts_model_R)

        # points of interest
        p1 = self.pnts_model.T[0]
        p2 = self.pnts_model.T[self.nb_pnts//2]
        p3 = self.pnts_model.T[-1]
        ux = [1, 0, 0]    # unit vector
        uy = [0, 1, 0]
        uz = [0, 0, 1]
        p1_tr = self.transform_pnts(self.T, p1)
        p2_tr = self.transform_pnts(self.T, p2)
        p3_tr = self.transform_pnts(self.T, p3)
        axes = [self.T[:3,:3].dot(ux), self.T[:3,:3].dot(uy), self.T[:3,:3].dot(uz)]
        img_left = self.draw_axes(img_left, p1_tr, axes, length=7, thickness=2)
        img_left = self.draw_axes(img_left, p2_tr, axes, length=7, thickness=2)
        img_left = self.draw_axes(img_left, p3_tr, axes, length=7, thickness=2)
        stacked = ImgUtils.stack_stereo_img(img_left, img_right, scale=0.7)
        cv2.imshow("", stacked)
        cv2.waitKey(1)
        return self.T, self.pnts_model

    def overlay_pnts(self, image, pnts, color=(0, 255, 0)):
        image_copy = image.copy()
        pnts = pnts.astype(int)
        if len(np.shape(image)) == 3:
            for p in pnts.T:
                cv2.circle(image_copy, (p[0], p[1]), 3, color, -1)
        elif len(np.shape(image)) == 2:
            image_copy[pnts[1, :], pnts[0, :]] = 255
        else:
            raise ValueError
        return image_copy

    def draw_axes(self, image, p_org, axes, length, thickness):
        ux, uy, uz = axes   # unit vectors
        pnt_org, _ = self.av.world2pixel(p_org, which='left')
        pnt_ux, _ = self.av.world2pixel(p_org + ux/np.linalg.norm(ux)/1000*length, which='left')
        pnt_uy, _ = self.av.world2pixel(p_org + uy/np.linalg.norm(uy)/1000*length, which='left')
        pnt_uz, _ = self.av.world2pixel(p_org + uz/np.linalg.norm(uz)/1000*length, which='left')

        pnt_org = pnt_org.astype(int).squeeze()
        pnt_ux = pnt_ux.astype(int).squeeze()
        pnt_uy = pnt_uy.astype(int).squeeze()
        pnt_uz = pnt_uz.astype(int).squeeze()

        img = image.copy()
        cv2.arrowedLine(img, tuple(pnt_org), tuple(pnt_ux), (0, 0, 255), thickness)     # x-axis
        cv2.arrowedLine(img, tuple(pnt_org), tuple(pnt_uy), (0, 255, 0), thickness)     # y-axis
        cv2.arrowedLine(img, tuple(pnt_org), tuple(pnt_uz), (255, 0, 0), thickness)     # z-axis
        return img

    # def visualize(self, iteration, error, X, Y, ax):
    #     plt.cla()
    #     ax.scatter(X[:, 0], X[:, 1], color='red', label='Target')
    #     ax.scatter(Y[:, 0], Y[:, 1], color='blue', label='Source')
    #     plt.text(0.87, 0.92, 'Iteration: {:d}\nQ: {:06.4f}'.format(
    #         iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
    #              fontsize='x-large')
    #     ax.legend(loc='upper left', fontsize='x-large')
    #     plt.draw()
    #     plt.pause(0.001)
    #
    #     # Create 3D plot
    #     # self.ax.cla()
    #     # self.ax.scatter(pnts_3D[0,:], pnts_3D[1,:], pnts_3D[2,:], marker='o')
    #     # self.ax.set_xlabel('X Label')
    #     # self.ax.set_ylabel('Y Label')
    #     # self.ax.set_zlabel('Z Label')
    #     # # self.ax.set_xlim3d(-0.1, 0.1)
    #     # # self.ax.set_ylim3d(-0.1, 0.1)
    #     # # self.ax.set_zlim3d(0.25, 0.4)
    #     # plt.draw()
    #     # plt.pause(0.001)
    #
    # def find_normals(self, pnts_3D, method='least_square'):   # A0 x + B0 y + C0 z = 1
    #     # least square problem
    #     x = pnts_3D[0, :].reshape(-1,1)
    #     y = pnts_3D[1, :].reshape(-1, 1)
    #     z = pnts_3D[2, :].reshape(-1, 1)
    #     A = np.concatenate((x, y, np.ones_like(x)), axis=1)
    #     b = -z
    #     if method=='least_square':
    #         c0, c1, c2 = np.linalg.lstsq(A, b, rcond=None)[0].squeeze()
    #     elif method=='RANSAC':
    #         ransac = linear_model.RANSACRegressor()
    #         ransac.fit(A, b)
    #         c0, c1, _ = ransac.estimator_.coef_[0]
    #         c2 = ransac.estimator_.intercept_[0]
    #     else:
    #         raise ValueError
    #     C0 = -1 / c2
    #     A0 = c0 * C0
    #     B0 = c1 * C0
    #     nz = [A0, B0, C0]
    #     nz = nz / np.linalg.norm(nz)
    #
    #     # if nz[2] < 0:
    #     #     nz *= -1
    #
    #     # Find nx, ny
    #     dist1 = distance.cdist(pnts_3D.T, pnts_3D.T)
    #     ind1 = np.unravel_index(np.argmax(dist1, axis=None), dist1.shape)
    #     nx = pnts_3D[:,ind1[0]] - pnts_3D[:,ind1[1]]
    #     nx = nx/np.linalg.norm(nx)
    #     # if nx[0] < 0:
    #     #     nx *= -1
    #     ny = np.cross(nz,nx)
    #
    #     # find origin
    #     p_mid = (pnts_3D[:, ind1[0]] + pnts_3D[:, ind1[1]]) / 2
    #     dist2 = distance.cdist(p_mid.reshape(1,-1), pnts_3D.T)
    #     ind2 = dist2.argmin()
    #     p_org = pnts_3D[:,ind2]
    #     p_org = pnts_3D[:, ind1[0]]
    #     return nx, ny, nz, p_org