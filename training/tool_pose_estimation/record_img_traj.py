from FLSpegtransfer.motion.dvrkArm import dvrkArm
from FLSpegtransfer.vision.AlliedVisionCapture import AlliedVisionCapture
import time
import numpy as np
import cv2


def stereo_img_display(img_left, img_right, scale):
    w = int(img_left.shape[1] * scale)
    h = int(img_left.shape[0] * scale)
    dim = (w, h)
    img_left_resized = cv2.resize(img_left.copy(), dim, interpolation=cv2.INTER_AREA)
    img_right_resized = cv2.resize(img_right.copy(), dim, interpolation=cv2.INTER_AREA)
    img_stacked = np.concatenate((img_left_resized, img_right_resized), axis=1)
    cv2.imshow("stereo_image", img_stacked)
    cv2.waitKey(1)


p1 = dvrkArm('/PSM1')
av = AlliedVisionCapture()

# capturing loop
time_curr = 0.0
time_st = time.time()
cnt = 0
print ("Recording started!")
while True:
    img_left, img_right = av.capture(which='rectified')
    if len(img_left) == 0 or len(img_right) == 0:
        time_curr = 0.0
        time_st = time.time()
    else:
        time_curr = time.time() - time_st
        joint1 = p1.get_current_joint(wait_callback=True)
        time.sleep(0.1)
        print("image" + str(int(cnt)), joint1, time_curr)

        # save data
        # cv2.imwrite("img_left" + str(int(cnt)) + ".png", img_left)
        # cv2.imwrite("img_right" + str(int(cnt)) + ".png", img_right)
        np.save("img_left" + str(int(cnt)), img_left)
        np.save("img_right" + str(int(cnt)), img_right)
        np.save("joint" + str(cnt), joint1)

        # Display
        stereo_img_display(img_left, img_right, scale=0.7)

        # Break the loop
        if time_curr > 400:
            break
        else:
            cnt += 1

print ("Recording terminated.")