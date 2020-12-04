import numpy as np
import cv2
from FLSpegtransfer.vision.AlliedVisionCapture import AlliedVisionCapture
from FLSpegtransfer.utils.ImgUtils import ImgUtils
import time

# define instance
av = AlliedVisionCapture()

# capturing loop
cnt = 0
while True:
    img_left, img_right = av.capture(which='rectified')
    if len(img_left) == 0 or len(img_right) == 0:
        pass
    else:
        np.save("img_left" + str(int(cnt)), img_left)
        np.save("img_right" + str(int(cnt)), img_right)
        cnt += 1

        # Display
        stacked = ImgUtils.stack_stereo_img(img_left, img_right, scale=0.7)
        cv2.imshow("", stacked)
        cv2.waitKey(1)
        time.sleep(0.1)

        # key = cv2.waitKey(1) & 0xFF
        # if key == ord('\r'):  # ENTER
        #     cv2.imwrite("img_left" + str(int(cnt)) + ".png", img_left)
        #     cv2.imwrite("img_right" + str(int(cnt)) + ".png", img_right)
        #     print("image" + str(int(cnt)) + " saved.")
        #     cnt += 1
        # # Display
        # stacked = ImgUtils.stack_stereo_img(img_left, img_right, scale=0.7)
        # cv2.imshow("", stacked)
        # cv2.waitKey(1)