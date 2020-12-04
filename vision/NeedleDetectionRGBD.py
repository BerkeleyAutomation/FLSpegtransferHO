from FLSpegtransfer.path import *
from FLSpegtransfer.vision.ZividUtils import ZividUtils
from sklearn import linear_model
import cv2
import numpy as np


class NeedleDetectionRGBD:
    def __init__(self):
        # thresholding value
        self.__lower_red = np.array([0 - 20, 60, 50])
        self.__upper_red = np.array([0 + 20, 255, 255])
        self.__lower_green = np.array([60 - 40, 100, 50])
        self.__upper_green = np.array([60 - 10, 255, 255])
        self.__lower_blue = np.array([120 - 20, 70, 30])
        self.__upper_blue = np.array([120 + 20, 255, 255])
        self.__lower_yellow = np.array([30 - 10, 130, 60])
        self.__upper_yellow = np.array([30 + 10, 255, 255])

    def mask_image(self, img_color, img_point, color, visualize=False):
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
        img_masked = cv2.inRange(img_hsv, hsv_range[0], hsv_range[1])

        # color masking
        con1 = (img_masked == 255)
        arg1 = np.argwhere(con1)
        pnt1 = img_point[con1]
        if len(arg1) < 500:
            mask = np.zeros_like(img_masked)
        else:
            # remove nan
            con2 = (~np.isnan(pnt1).any(axis=1))
            arg2 = np.argwhere(con2)

            # creat mask where color & depth conditions hold
            arg_mask = np.squeeze(arg1[arg2])
            mask = np.zeros_like(img_masked)
            mask[arg_mask[:, 0], arg_mask[:, 1]] = 255
        return mask

    def find_needle(self, img_color, img_point, color, visualize=False):
        if visualize:
            cv2.imshow("", img_color)
            cv2.waitKey(0)

        # color masking
        masked = self.mask_image(img_color, img_point, color, visualize)
        if visualize:
            cv2.imshow("", masked)
            cv2.waitKey(0)

        # Find contours to threshold
        cnts, _ = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        cnts_needle = [c for c in cnts if cv2.contourArea(c) > 60]  # threshold by contour size
        if visualize:
            img_color_copy = img_color.copy()
            cv2.drawContours(img_color_copy, cnts_needle, -1, (0, 255, 255), 1)
            cv2.imshow("", img_color_copy)
            cv2.waitKey(0)

        # Get the pixel coordinates inside the contour
        infilled = np.zeros(np.shape(img_color), np.uint8)
        cv2.drawContours(infilled, cnts_needle, -1, (255, 255, 255), -1)
        infilled = cv2.cvtColor(infilled, cv2.COLOR_BGR2GRAY)
        needle_masked = cv2.bitwise_and(masked, masked, mask=infilled)
        if visualize:
            cv2.imshow("", needle_masked)
            cv2.waitKey(0)

        # Get the points
        args = np.argwhere(needle_masked==255)
        x = args[:, 1].reshape(-1, 1)
        y = args[:, 0].reshape(-1, 1)
        argmax = np.argmax(x)
        pn = args[argmax]
        if visualize:
            img_color_copy = np.copy(img_color)
            cv2.circle(img_color_copy, (int(pn[1]), int(pn[0])), 3, (0, 255, 255), -1)
            cv2.imshow("", img_color_copy)
            cv2.waitKey(0)
        return img_point[pn[0], pn[1]]