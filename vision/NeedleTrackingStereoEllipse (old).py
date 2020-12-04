from FLSpegtransfer.vision.AlliedVisionUtils import AlliedVisionUtils
import FLSpegtransfer.utils.CmnUtil as U
from FLSpegtransfer.utils.ImgUtils import ImgUtils
from FLSpegtransfer.vision.PCLRegistration import PCLRegistration
from sklearn import linear_model
from skimage.morphology import skeletonize
from scipy.spatial import distance
from pycpd import AffineRegistration
import time
import cv2
import numpy as np
from mayavi import mlab
import open3d as o3d
import copy
from FLSpegtransfer.utils.filters import LPF
from FLSpegtransfer.utils.plot import *
from probreg import features
from probreg import callbacks
from probreg import l2dist_regs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from skimage.transform import rescale, resize, downscale_local_mean
from functools import partial
from scipy import ndimage

class NeedleTrackingStereoEllipse:
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
        self.nb_pnts = 100
        self.T = np.identity(4)
        # self.filter = LPF(fc=3, fs=100, order=2, nb_axis=2)

        fig = plt.figure()
        self.ax = fig.add_subplot(111, projection='3d')
        plt.ion()
        plt.show()

        # instances
        self.av = AlliedVisionUtils()
        self.pnts_model = self.sample_3Dpoints(nb_pnts=self.nb_pnts, radius=.017, st_angle=np.deg2rad(0+10), ed_angle=np.deg2rad(180-10))

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
        z = np.ones_like(x)*0.05    # 30 cm far from lens
        p = np.concatenate((x, y, z), axis=0)
        return p  # dim(p) = 3xn

    @classmethod
    def fit_ellipse(cls, x, y, method='RANSAC', w=None):
        if w is None:
            w = []
        if method=='least_square':
            A = np.concatenate((x**2, x*y, y**2, x, y), axis=1)
            b = np.ones_like(x)

            # Modify A,b for weighted least squares
            if len(w) == len(x):
                W = np.diag(w)
                A = np.dot(W, A)
                b = np.dot(W, b)

            # Solve by method of least squares
            c = np.linalg.lstsq(A, b, rcond=None)[0].squeeze()

            # Get circle parameters from solution
            A0 = c[0]
            B0 = c[1] / 2
            C0 = c[2]
            D0 = c[3] / 2
            E0 = c[4] / 2
        elif method=='RANSAC':
            A = np.concatenate((x**2, x*y, y**2, x), axis=1)
            b = -2*y
            ransac = linear_model.RANSACRegressor()
            ransac.fit(A, b)
            c0, c1, c2, c3 = ransac.estimator_.coef_[0]
            c4 = ransac.estimator_.intercept_[0]
            E0 = -1/c4
            A0 = c0*E0
            B0 = c1*E0/2
            C0 = c2*E0
            D0 = c3*E0/2
        else:
            raise ValueError

        # center of ellipse
        cx = (C0*D0 - B0*E0)/(B0**2 - A0*C0)
        cy = (A0*E0 - B0*D0)/(B0**2 - A0*C0)
        temp = 1.0 - A0*cx**2 - 2.0*B0*cx*cy - C0*cy**2 - 2.0*D0*cx - 2.0*E0*cy
        A1 = A0/temp
        B1 = B0/temp
        C1 = C0/temp

        # rotating angle of ellipse
        M = A1**2 + C1**2 + 4*B1**2 - 2*A1*C1
        theta = np.arcsin(np.sqrt((-(C1-A1)*np.sqrt(M) + M)/(2*M)))

        # length of axis of ellipse
        a = np.sqrt(1.0/(A1*np.cos(theta)**2 + 2*B1*np.cos(theta)*np.sin(theta)+C1*np.sin(theta)**2))
        b = np.sqrt(1.0/(A1*np.sin(theta)**2 - 2*B1*np.sin(theta)*np.cos(theta)+C1*np.cos(theta)**2))
        return cx,cy, a,b, theta

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
        # Gaussian Blur
        # img_masked = cv2.GaussianBlur(img_masked, (3, 3), sigmaX=10, sigmaY=10)

        # Find contours
        cnts, _ = cv2.findContours(img_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        cnts_needle = [c for c in cnts if cv2.contourArea(c) > 80]  # threshold by contour size

        # # smooth contours
        # cnts_needle_smooth = []
        # for cnt in cnts_needle:
        #     cnt = cnt.squeeze(axis=1)
        #     cnt = np.tile(cnt, (5,1))
        #     kernel = (1,10)
        #     cnt = cv2.blur(cnt, kernel)   # blur (same as moving average filter)
        #     cnt = self.filter.filter(cnt)   # LPF
        #     cnt = cnt.astype(int)
        #     # plot_filter(cnt[100:], filtered[100:], show_window=True)
        #     h = np.shape(cnt)[0]
        #     cnts_needle_smooth.append(cnt[int(h*3/5):int(h*4/5), :])
        # if visualize:
        #     img_color_copy = img_color.copy()
        #     cv2.drawContours(img_color_copy, cnts_needle_smooth, -1, (0, 255, 0), 1)
        #     cv2.imshow("", img_color_copy)
        #     cv2.waitKey(0)

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
    def clean_pnts(self, img_nd_masked, img_nd_skeleton):
        pnts_skeleton = self.get_coordinate(img_nd_masked)
        # pnts_skeleton = self.get_coordinate(img_nd_skeleton)
        (cx, cy), (a, b), angle = cv2.fitEllipse(pnts_skeleton.T)  # Fit ellipse (OpenCV method)
        a = a / 2
        b = b / 2
        pnts_sample = self.sample_2Dpoints(cx, cy, a, b, nb_pnts=self.nb_pnts)
        pnts_rotated = self.rotate_points(cx, cy, pnts_sample, np.deg2rad(angle))
        img_nd_skeleton_cleaned = cv2.bitwise_and(img_nd_masked, self.overlay_pnts2(np.zeros_like(img_nd_masked), pnts_rotated))
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

        if visualize:
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            src.paint_uniform_color([1, 0, 0])
            tgt.paint_uniform_color([0.8, 0.8, 0.8])
            vis.add_geometry(src)
            vis.add_geometry(tgt)
            vis.get_view_control().rotate(1000.0, 150)
            for _ in range(10):
                vis.get_view_control().scale(0.8)
            icp_iteration = 100
            threshold = [20, 10, 5, 2, 1, 0.5, 0.2]
            for i in range(len(threshold)):
                for j in range(icp_iteration):
                    reg_p2p = o3d.registration.registration_icp(src, tgt, threshold[i], np.identity(4),
                                                                o3d.registration.TransformationEstimationPointToPoint(),
                                                                o3d.registration.ICPConvergenceCriteria(max_iteration=1))
                    src.transform(reg_p2p.transformation)
                    T = copy.deepcopy(reg_p2p.transformation).dot(T)
                    vis.update_geometry(src)
                    vis.update_geometry(tgt)
                    vis.poll_events()
                    vis.update_renderer()
                    if save_image:
                        vis.capture_screen_image("image_%04d.jpg" % j)
            vis.run()
        else:
            # Point-to-point ICP
            # threshold = [20, 10, 5, 2, 1, 0.5, 0.2]
            # for i in range(len(threshold)):
            threshold = 0.1
            reg_p2p = o3d.registration.registration_icp(src, tgt, threshold, T0,
                                                        o3d.registration.TransformationEstimationPointToPoint())
            # src.transform(reg_p2p.transformation)
            # T = copy.deepcopy(reg_p2p.transformation).dot(T)
            T = copy.deepcopy(reg_p2p.transformation)
        return T

    def find_normals(self, pnts_3D, method='least_square'):   # A0 x + B0 y + C0 z = 1
        # least square problem
        x = pnts_3D[0, :].reshape(-1,1)
        y = pnts_3D[1, :].reshape(-1, 1)
        z = pnts_3D[2, :].reshape(-1, 1)
        A = np.concatenate((x, y, np.ones_like(x)), axis=1)
        b = -z
        if method=='least_square':
            c0, c1, c2 = np.linalg.lstsq(A, b, rcond=None)[0].squeeze()
        elif method=='RANSAC':
            ransac = linear_model.RANSACRegressor()
            ransac.fit(A, b)
            c0, c1, _ = ransac.estimator_.coef_[0]
            c2 = ransac.estimator_.intercept_[0]
        else:
            raise ValueError
        C0 = -1 / c2
        A0 = c0 * C0
        B0 = c1 * C0
        nz = [A0, B0, C0]
        nz = nz / np.linalg.norm(nz)

        # if nz[2] < 0:
        #     nz *= -1

        # Find nx, ny
        dist1 = distance.cdist(pnts_3D.T, pnts_3D.T)
        ind1 = np.unravel_index(np.argmax(dist1, axis=None), dist1.shape)
        nx = pnts_3D[:,ind1[0]] - pnts_3D[:,ind1[1]]
        nx = nx/np.linalg.norm(nx)
        # if nx[0] < 0:
        #     nx *= -1
        ny = np.cross(nz,nx)

        # find origin
        p_mid = (pnts_3D[:, ind1[0]] + pnts_3D[:, ind1[1]]) / 2
        dist2 = distance.cdist(p_mid.reshape(1,-1), pnts_3D.T)
        ind2 = dist2.argmin()
        p_org = pnts_3D[:,ind2]
        p_org = pnts_3D[:, ind1[0]]
        return nx, ny, nz, p_org

    def find_needle2D(self, img_color, color, visualize=False):
        if visualize:
            cv2.imshow("", img_color)
            cv2.waitKey(0)

        # color masking
        img_masked = self.mask_color(img_color, color, visualize)
        needle_masked, cnts_needle = self.mask_needle(img_color, img_masked, visualize)
        # needle_skeleton = self.skeletonize(needle_masked, visualize)
        needle_skeleton = []
        # if visualize:
        #     overlay_skeleton = cv2.add(img_color, cv2.cvtColor(needle_skeleton, cv2.COLOR_GRAY2BGR))
        #     cv2.imshow("", overlay_skeleton)
        #     cv2.waitKey(0)
        return needle_masked, needle_skeleton

    def transform_affine(self, pnts_L, pose_affine):
        sx, hx, tx = pose_affine
        T_aff = np.array([[sx, hx], [0, 1]])
        pnts_L_tr = T_aff.dot(pnts_L) + np.array([[tx], [0]])
        return pnts_L_tr

    def update_affine(self, pnts_L, pnts_R):
        dist = distance.cdist(pnts_L.T, pnts_R.T)
        args_pair = np.argmin(dist, axis=1)

        # least-square fit
        A = np.concatenate((pnts_L.T, np.ones((len(pnts_L.T), 1))), axis=1)
        b = pnts_R.T[args_pair][:,0]

        # Solve by method of least squares
        c = np.linalg.lstsq(A, b, rcond=None)[0].squeeze()
        sx = c[0]
        hx = c[1]
        tx = c[2]
        pose_affine = [sx, hx, tx]
        return pose_affine

    def error(self, imgs_LR, pnts_LR, pose_affine):
        sx, hx, tx = pose_affine
        img_L, img_R = imgs_LR
        pnts_L, pnts_R = pnts_LR

        # # crop (y_min, y_max) and digitize image array
        # args_L = np.argwhere(img_L==255)
        # y_max = np.max(args_L[:, 0])
        # y_min = np.min(args_L[:, 0])
        # dh = 8
        # delta = (y_max-y_min)//dh*(dh+1)
        # img_L_cropped = img_L[y_min:y_min+delta, :]
        # img_split = np.split(img_L_cropped, dh, axis=0)
        # img_split = np.array(img_split)
        #
        # args = np.argwhere(img_split==255)
        # ind = np.nonzero(np.diff(args[:, 0]))[0] + 1      # Find the split indices
        # ind = np.insert(ind, 0, 0)
        # c0 = np.arange(ind.size)  # the 1st column
        # n = np.insert(ind, len(ind), len(args))
        # c1 = np.add.reduceat(args[:, 2], ind)/np.diff(n)  # the 2nd column
        # result = np.c_[c0, c1]

        T_aff = np.array([[sx, hx],[0, 1]])
        pnts_L_tr = T_aff.dot(pnts_L) + np.array([[tx], [0]])
        errors = distance.cdist(pnts_L_tr.T, pnts_R.T).min(axis=1).sum()
        return errors

    def error_grad(self, imgs_LR, pnts_LR, pose_affine):
        sx, hx, tx = pose_affine
        # variation size
        dsx = 0.01
        dhx = 0.01
        dtx = 0.01
        derr_sx = (self.error(imgs_LR, pnts_LR, [sx+dsx, hx, tx]) - self.error(imgs_LR, pnts_LR, [sx-dsx, hx, tx]))/(2*dsx)
        derr_hx = (self.error(imgs_LR, pnts_LR, [sx, hx+dhx, tx]) - self.error(imgs_LR, pnts_LR, [sx, hx-dhx, tx]))/(2*dhx)
        derr_tx = (self.error(imgs_LR, pnts_LR, [sx, hx, tx+dtx]) - self.error(imgs_LR, pnts_LR, [sx, hx, tx-dtx]))/(2*dtx)
        return np.array([derr_sx, derr_hx, derr_tx])

    def find_needle3D(self, img_left, img_right, color, visualize=False):
        img_nd_masked_L, img_nd_skeleton_L = self.find_needle2D(img_left, color, visualize=False)
        img_nd_masked_R, img_nd_skeleton_R = self.find_needle2D(img_right, color, visualize=False)

        # downsample
        downsampled = np.zeros_like(img_nd_masked_L)
        downsampled[::5, ::5] = 255
        img_nd_masked_L_ds = np.bitwise_and(img_nd_masked_L, downsampled)
        img_nd_masked_R_ds = np.bitwise_and(img_nd_masked_R, downsampled)

        # if visualize:
        # stacked = ImgUtils.stack_stereo_img(img_nd_masked_L_ds, img_nd_masked_R_ds, scale=0.7)
        # cv2.imshow("", stacked)
        # cv2.waitKey(0)

        # get affine transform
        pnts_needle_L = self.get_coordinate(img_nd_masked_L)
        pnts_needle_R = self.get_coordinate(img_nd_masked_R)

        # clean image points
        pnts_cln_L, ellipse_L = self.clean_pnts(img_nd_masked_L, img_nd_masked_L)
        pnts_cln_R, ellipse_R = self.clean_pnts(img_nd_masked_R, img_nd_masked_R)

        # pts1 = pnts_cln_L.T
        # pts2 = pnts_cln_R.T
        # mean1 = pts1.mean(axis=0)
        # mean2 = pts2.mean(axis=0)
        # t = (mean2-mean1)
        # pnts_cln_L_tr = pnts_cln_L.copy() + t.reshape(2,-1)
        # st = time.time()
        # while True:
        #     # imgs_LR = (img_nd_masked_L, img_nd_masked_R)
        #     # pnts_LR = (pnts_needle_L, pnts_needle_R)
        #     pose_affine = self.update_affine(pnts_cln_L_tr, pnts_cln_R)
        #     pnts_cln_L_tr = self.transform_affine(pnts_cln_L_tr, pose_affine)
        #     if np.isclose(pose_affine, np.array([1.0, 0.0, 0.0]), rtol=1e-03, atol=1e-06).all():
        #         break
        #     img_left_copy = img_left.copy()
        #     img_right_copy = img_right.copy()
        #     img_left_copy = self.overlay_pnts(img_left_copy, pnts_cln_L, pnts_cln_L)
        #     img_right_copy = self.overlay_pnts(img_right_copy, pnts_cln_L_tr, pnts_cln_R)
        #     # img_left = self.draw_axes(img_left, p_org, [nx, ny, nz], length=10, thickness=2)
        #     stacked = ImgUtils.stack_stereo_img(img_left_copy, img_right_copy, scale=0.7)
        #     cv2.imshow("", stacked)
        #     cv2.waitKey(0)
        # print (time.time() - st)

        # cx, cy, a, b, angle = ellipse_L
        # cv2.ellipse(img_left, (int(cx), int(cy)), (int(a), int(b)), angle, 0, 360, (0, 255, 0), 2)
        # cx, cy, a, b, angle = ellipse_R
        # cv2.ellipse(img_right, (int(cx), int(cy)), (int(a), int(b)), angle, 0, 360, (0, 255, 0), 2)
        # stacked = ImgUtils.stack_stereo_img(img_left, img_right, scale=0.7)
        # cv2.imshow("", stacked)
        # cv2.waitKey(1)

        # if visualize:
        # stacked = ImgUtils.stack_stereo_img(img_nd_masked_L, img_nd_masked_R, scale=0.7)
        # cv2.imshow("", stacked)
        # cv2.waitKey(1)

        # obtain affine transform
        # fig = plt.figure()
        # fig.add_axes([0, 0, 1, 1])
        # callback = partial(self.visualize, ax=fig.axes[0])
        st = time.time()
        reg = AffineRegistration(**{'X': pnts_cln_R.T, 'Y': pnts_cln_L.T})
        TY, (B, t) = reg.register()     # affine transform matrix M = B.T
        # TY, (B, t) = reg.register(callback)  # affine transform matrix M = B.T
        print(B, t)
        # print (time.time() - st)
        # plt.show()

        pnts_cln_R_new = TY.T
        # pnts_cln_R_new = B.T.dot(pnts_cln_L) + t.reshape(2,1)

        img_left_copy = img_left.copy()
        img_right_copy = img_right.copy()
        img_left_copy = self.overlay_pnts(img_left_copy, pnts_cln_L, pnts_cln_L)
        img_right_copy = self.overlay_pnts(img_right_copy, pnts_cln_R_new, pnts_cln_R)
        # img_left = self.draw_axes(img_left, p_org, [nx, ny, nz], length=10, thickness=2)
        stacked = ImgUtils.stack_stereo_img(img_left_copy, img_right_copy, scale=0.7)
        cv2.imshow("", stacked)
        cv2.waitKey(1)

        # if visualize:
        #     img_left = self.overlay_pnts(img_left, pnts_cln_L)
        #     img_right = self.overlay_pnts(img_right, pnts_cln_R)
        #     stacked = ImgUtils.stack_stereo_img(img_left, img_right, scale=0.7)
        #     cv2.imshow("", stacked)
        #     cv2.imwrite("test.png", stacked)
        #     cv2.waitKey(0)

        # reconstruct 3D points
        pnts_3D = self.av.pixel2world(pnts_cln_L, pnts_cln_R_new)
        # img_left = self.overlay_pnts(img_left, pnts_cln_L)
        # img_right = self.overlay_pnts(img_right, pnts_cln_R)
        # stacked = ImgUtils.stack_stereo_img(img_left, img_right, scale=0.7)
        # cv2.imshow("", stacked)
        # cv2.waitKey(1)
        #
        # # Create 3D plot
        self.ax.cla()
        self.ax.scatter(pnts_3D[0,:], pnts_3D[1,:], pnts_3D[2,:], marker='o')
        self.ax.set_xlabel('X Label')
        self.ax.set_ylabel('Y Label')
        self.ax.set_zlabel('Z Label')
        self.ax.set_xlim3d(-0.03, 0.0)
        self.ax.set_ylim3d(0.02, 0.03)
        self.ax.set_zlim3d(0.275, 0.2875)
        plt.draw()
        plt.pause(0.001)

        # # make a registration of needle model to 3D points
        # pcl_source = o3d.geometry.PointCloud()
        # pcl_target = o3d.geometry.PointCloud()
        # pcl_source.points = o3d.utility.Vector3dVector(self.pnts_model.T)
        # pcl_target.points = o3d.utility.Vector3dVector(pnts_3D.T)
        # self.T = self.registration(pcl_source, pcl_target, T0=self.T, use_svr=False, save_image=False, visualize=False)
        # # print (self.T)
        # T = self.T
        # # T = np.linalg.inv(T)  # transform from model to the reconstructed needle
        # pnts_model_tr = T[:3, :3].dot(self.pnts_model).T + T[:3, -1].T  # transformed mask points
        # pnts_model_tr = pnts_model_tr.T

        # 2D image points projected on left/right image frame
        # pnts_model_L, _ = self.av.world2pixel(pnts_model_tr, which='left')
        # pnts_model_R, _ = self.av.world2pixel(pnts_model_tr, which='right')

        # diff = pnts_cln_R_new - pnts_model_R
        # import pdb; pdb.set_trace()

        # fit plane
        # nx, ny, nz, p_org = self.find_normals(pnts_3D, method='RANSAC')
        # if visualize:
        # img_left = self.overlay_pnts(img_left, pnts_model_L)
        # img_right = self.overlay_pnts(img_right, pnts_model_R)
        # img_left = self.overlay_pnts(img_left, pnts_cln_L)
        # img_right = self.overlay_pnts(img_right, pnts_cln_R_new)
        # img_left = self.draw_axes(img_left, p_org, [nx,ny,nz], length=10, thickness=2)
        # self.plot3d(pnts_3D, p_org.reshape(-1,1))
        # stacked = ImgUtils.stack_stereo_img(img_left, img_right, scale=0.7)
        # cv2.imshow("", stacked)
        # cv2.waitKey(1)
        return [], []

    def overlay_pnts2(self, image, pnts):
        try:
            image_copy = image.copy()
            pnts = pnts.astype(int)
            if len(np.shape(image)) == 3:
                for p in pnts.T:
                    cv2.circle(image_copy, (p[0], p[1]), 2, (0, 255, 0), -1)
                # image_copy[pnts[1, :], pnts[0, :]] = [0, 255, 0]
            elif len(np.shape(image)) == 2:
                image_copy[pnts[1, :], pnts[0, :]] = 255
            else:
                raise ValueError
            return image_copy
        except:
            pass

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

    def visualize(self, iteration, error, X, Y, ax):
        plt.cla()
        ax.scatter(X[:, 0], X[:, 1], color='red', label='Target')
        ax.scatter(Y[:, 0], Y[:, 1], color='blue', label='Source')
        plt.text(0.87, 0.92, 'Iteration: {:d}\nQ: {:06.4f}'.format(
            iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
                 fontsize='x-large')
        ax.legend(loc='upper left', fontsize='x-large')
        plt.draw()
        plt.pause(0.001)

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
        # cv2.arrowedLine(img, tuple(pnt_org), tuple(pnt_nx), (0, 0, 255), thickness)     # x-axis
        # cv2.arrowedLine(img, tuple(pnt_org), tuple(pnt_ny), (0, 255, 0), thickness)     # y-axis
        cv2.arrowedLine(img, tuple(pnt_org), tuple(pnt_nz), (255, 0, 0), thickness)     # z-axis
        return img

    def plot3d(self, pnts, pnts2):
        pnts = np.array(pnts.T)
        pnts2 = np.array(pnts2.T)
        mlab.figure("", fgcolor=(0., 0., 0.), bgcolor=(1, 1, 1), size=(1200, 900))  # black background
        mlab.points3d(pnts[:, 0], pnts[:, 1], pnts[:, 2], color=(0.0, 0.0, 1.0), scale_factor=0.001)
        mlab.points3d(pnts2[:, 0], pnts2[:, 1], pnts2[:, 2], color=(1.0, 0.0, 0.0), scale_factor=0.001)
        mlab.axes(xlabel='x', ylabel='y', zlabel='z', z_axis_visibility=False)
        mlab.orientation_axes()
        mlab.outline(color=(.7, .7, .7))
        mlab.view(azimuth=180, elevation=180)
        mlab.show()