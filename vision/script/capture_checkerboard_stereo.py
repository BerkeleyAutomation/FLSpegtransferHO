import numpy as np
import cv2
from FLSpegtransfer.vision.AlliedVisionCapture import AlliedVisionCapture


def stereo_img_display(img_left, img_right, scale):
    w = int(img_left.shape[1] * scale)
    h = int(img_left.shape[0] * scale)
    dim = (w, h)
    img_left_resized = cv2.resize(img_left, dim, interpolation=cv2.INTER_AREA)
    img_right_resized = cv2.resize(img_right, dim, interpolation=cv2.INTER_AREA)
    img_stacked = np.concatenate((img_left_resized, img_right_resized), axis=1)
    cv2.imshow("stereo_image", img_stacked)
    cv2.waitKey(1)


# define instance
av = AlliedVisionCapture()

# grid information
row = 11
col = 7

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# capturing loop
cnt = 0
while True:
    img_left, img_right = av.capture(which='original')
    if len(img_left) == 0 or len(img_right) == 0:
        pass
    else:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('\r'):  # ENTER
            gray_left = cv2.cvtColor(img_left, cv2.COLOR_RGB2GRAY)
            gray_right = cv2.cvtColor(img_right, cv2.COLOR_RGB2GRAY)

            # Find the chess board corners
            ret_left, corners_left = cv2.findChessboardCorners(gray_left, (col, row), None)
            ret_right, corners_right = cv2.findChessboardCorners(gray_right, (col, row), None)

            # If found, add object points, image points (after refining them)
            print (ret_left, ret_right)
            if ret_left == True and ret_right == True:
                corners_ref_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
                corners_ref_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)

                # Draw and display the corners
                img_left_copy = cv2.drawChessboardCorners(img_left.copy(), (col, row), corners_ref_left, ret_left)
                img_right_copy = cv2.drawChessboardCorners(img_right.copy(), (col, row), corners_ref_right, ret_right)
                stereo_img_display(img_left_copy, img_right_copy, scale=0.7)
                cv2.waitKey(1)

                cv2.imwrite("img_left" + str(int(cnt)) + ".png", img_left)
                cv2.imwrite("img_right" + str(int(cnt)) + ".png", img_right)
                print("image" + str(int(cnt)) + " saved.")
                cnt += 1

        # Display
        stereo_img_display(img_left, img_right, scale=0.7)