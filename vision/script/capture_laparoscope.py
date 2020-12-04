from FLSpegtransfer.vision.Laparoscope import Laparoscope
import cv2
import numpy as np

laparo = Laparoscope()
cnt = 0
while True:
    img_left = laparo.img_left
    img_right = laparo.img_right
    if len(img_left) == 0 or len(img_right) == 0 or np.shape(img_left) != (1080, 1920, 3) or np.shape(img_right) != (1080, 1920, 3):
        pass
    else:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('\r'):  # ENTER
            # np.save("img_left", img_left)
            # np.save("img_right", img_right)
            cv2.imwrite("img_left" + str(int(cnt))+".png", img_left)
            cv2.imwrite("img_right" + str(int(cnt)) + ".png", img_right)
            print ("image" + str(int(cnt)) + " saved.")
            cnt += 1

        cv2.imshow("Left image", img_left)
        cv2.imshow("Right image", img_right)
        cv2.waitKey(1)