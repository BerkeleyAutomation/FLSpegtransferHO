from FLSpegtransfer.path import *
import numpy as np
import cv2


class AlliedVisionUtils:
    def __init__(self):
        # camera matrices
        path = 'calibration_files/alliedvision/'
        self.K = np.load(root + path + 'K.npy')
        self.R = np.load(root + path + 'R.npy')
        self.T = np.load(root + path + 'T.npy')
        self.Q = np.load(root + path + 'Q.npy')
        self.P1 = np.load(root + path + 'P1.npy')
        self.P2 = np.load(root + path + 'P2.npy')

        self.fx = self.K[:, 0, 0]
        self.fy = self.K[:, 1, 1]
        self.cx = self.K[:, 0, 2]
        self.cy = self.K[:, 1, 2]
        self.mapx1 = np.load(root+path+"mapx1.npy")
        self.mapy1 = np.load(root+path+"mapy1.npy")
        self.mapx2 = np.load(root+path+"mapx2.npy")
        self.mapy2 = np.load(root+path+"mapy2.npy")

    def rectify(self, img_left, img_right):
        rectified_left = cv2.remap(img_left, self.mapx1, self.mapy1, cv2.INTER_LINEAR)
        rectified_right = cv2.remap(img_right, self.mapx2, self.mapy2, cv2.INTER_LINEAR)
        return rectified_left, rectified_right

    def pixel2world(self, pnts_L, pnts_R):  # for rectified image
        disp = (pnts_L[0,:] - pnts_R[0,:]).reshape(1,-1)
        pnts_homo = np.concatenate((pnts_L, disp, np.ones_like(disp)), axis=0)  # [x, y, disparity, 1].T
        P = self.Q.dot(pnts_homo)
        X = (P[0,:] / P[3,:]).reshape(1,-1)
        Y = (P[1,:] / P[3,:]).reshape(1,-1)
        Z = (P[2,:] / P[3,:]).reshape(1,-1)
        # f = self.Q[2, 3]
        # R = (pnts_L[2,:] + pnts_R[2,:]) / 2 * Z.reshape(1,-1) / f
        pnts_3D = np.concatenate((X, Y, Z), axis=0)
        return pnts_3D  # [X,Y,Z]

    # def pixel2world(self, pl, pr):  # for rectified image
    #     pixel_homo = np.array([pl[0], pl[1], pl[0] - pr[0], 1]).T  # [x, y, disparity, 1].T
    #     P = self.Q.dot(pixel_homo)
    #     X = P[0] / P[3]
    #     Y = P[1] / P[3]
    #     Z = P[2] / P[3]
    #     f = self.Q[2, 3]
    #     R = (pl[2] + pr[2]) / 2 * Z / f
    #     return X, Y, Z, R

    def world2pixel(self, P, R=0, which='left'):  # for rectified image
        P = np.array(P).reshape(3,-1)
        ones = np.ones((1, np.shape(P)[1]))
        P = np.concatenate((P,ones), axis=0)        # P = [x, y, z, 1].T, dim(P) = (4, n)
        if which == 'left':
            p = self.P1.dot(P)
            fx = self.P1[0, 0]
            fy = self.P1[1, 1]
            x = (p[0, :] / p[2, :]).reshape(1, -1)
            y = (p[1, :] / p[2, :]).reshape(1, -1)
        elif which == 'right':
            p = self.P2.dot(P)
            fx = self.P2[0, 0]
            fy = self.P2[1, 1]
            x = (p[0, :] / p[2, :]).reshape(1, -1)
            y = (p[1, :] / p[2, :]).reshape(1, -1)    # rectification error (y-offset)
        else:
            raise ValueError
        p = np.concatenate((x,y), axis=0)
        if R==0:
            r = 0
        else:
            Z = P[3,:]
            r = (fx + fy) / 2 * R / Z
        return p, r     # p = [x,y]

if __name__ == "__main__":
    av_util = AlliedVisionUtils()