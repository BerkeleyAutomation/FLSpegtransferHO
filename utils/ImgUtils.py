import sys
for p in sys.path:
    if p == '/opt/ros/kinetic/lib/python2.7/dist-packages':
        sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np


class ImgUtils():

    @classmethod
    def transform_img(cls, img, rot_center, angle_deg, tx, ty):  # angle is positive in counter-clockwise
        R = cv2.getRotationMatrix2D((rot_center[0], rot_center[1]), angle_deg, 1)
        t = np.float32([[1, 0, tx], [0, 1, ty]])
        rotated = cv2.warpAffine(img, R, (img.shape[0], img.shape[1]))
        transformed = cv2.warpAffine(rotated, t, (img.shape[0], img.shape[1]))
        return transformed

    @classmethod
    def transform_pnts(cls, pnts, rot_center, angle_deg, tx, ty):
        pnts = np.array(pnts)
        R = cv2.getRotationMatrix2D((0, 0), angle_deg, 1)[:, :2]
        t = np.array([tx, ty])
        new_pnts = [R.dot(p - rot_center) + t + rot_center for p in pnts]
        return new_pnts


    # def downsample_naive(self, img, downsample_factor):
    #     """
    #     Naively downsamples image without LPF.
    #     """
    #     new_img = img.copy()
    #     new_img = new_img[::downsample_factor]
    #     new_img = new_img[:, ::downsample_factor]
    #     return new_img
    @classmethod
    def stack_stereo_img(cls, img1, img2, scale):
        w = int(img1.shape[1] * scale)
        h = int(img1.shape[0] * scale)
        dim = (w, h)
        img_resized1 = cv2.resize(img1, dim, interpolation=cv2.INTER_AREA)
        img_resized2 = cv2.resize(img2, dim, interpolation=cv2.INTER_AREA)
        img_stacked = np.concatenate((img_resized1, img_resized2), axis=1)
        return img_stacked

    @classmethod
    def compare_rectified_img(cls, img1, img2, scale, line_gap=30):
        img_stacked = ImgUtils.stack_stereo_img(img1, img2, scale)
        # draw horizontal lines every 25 px accross the side by side image
        for i in range(0, img_stacked.shape[0], line_gap):
            cv2.line(img_stacked, (0, i), (img_stacked.shape[1], i), (255, 0, 0))

        # for i in range(0, img_stacked.shape[1], line_gap):
        #     cv2.line(img_stacked, (i, 0), (i, img_stacked.shape[0]), (255, 0, 0))
        return img_stacked