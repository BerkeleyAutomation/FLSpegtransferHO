import cv2
print (cv2.__version__)
import numpy as np
from matplotlib import pyplot as plt
from FLSpegtransfer.vision.AlliedVisionCapture import AlliedVisionCapture
from FLSpegtransfer.utils.ImgUtils import ImgUtils
root = '/home/hwangmh/pycharmprojects/FLSpegtransfer/'


# define instance
av = AlliedVisionCapture()

# load images to rectify
# path = "calibration_alliedvision/"
# filename_left = 'img_left0.png'
# filename_right = 'img_right0.png'
# img1 = cv2.imread(path+filename_left, 0)
# img2 = cv2.imread(path+filename_right, 0)

# load mapping variables for stereo cameras
path = 'calibration_files/alliedvision/'
mapx1 = np.load(root+path+"mapx1.npy")
mapy1 = np.load(root+path+"mapy1.npy")
mapx2 = np.load(root+path+"mapx2.npy")
mapy2 = np.load(root+path+"mapy2.npy")

# capturing loop
cnt = 0
while True:
    img_left, img_right = av.capture()
    if len(img_left) == 0 or len(img_right) == 0:
        pass
    else:
        # remap images
        img_rect1 = cv2.remap(img_left, mapx1, mapy1, cv2.INTER_LINEAR)
        img_rect2 = cv2.remap(img_right, mapx2, mapy2, cv2.INTER_LINEAR)
        img_compared = ImgUtils.compare_rectified_img(img_rect1, img_rect2, scale=0.7, line_gap=40)
        cv2.imshow("stacked", img_compared)
        cv2.waitKey(0)
        # key = cv2.waitKey(1) & 0xFF
        # if key == ord('\r'):  # ENTER
        #     break

        # create disparity map
        window_size = 3
        min_disp = 16
        num_disp = 112-min_disp
        stereo = cv2.StereoSGBM_create(minDisparity=min_disp, numDisparities=num_disp, blockSize=16,
                                      P1=2144,  # 8*3*window_size**2,
                                      P2=1117,  # 32*3*window_size**2,
                                      disp12MaxDiff=10,
                                      uniquenessRatio=10,
                                      speckleWindowSize=10,  # 100, #set to 0 to disable speckle filtering
                                      speckleRange = 32)

        print('computing disparity...')
        img_gray1 = cv2.cvtColor(img_rect1, cv2.COLOR_BGR2GRAY)
        img_gray2 = cv2.cvtColor(img_rect2, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite("img_gray1.pgm", img_gray1)
        # cv2.imwrite("img_gray2.pgm", img_gray2)

        disparity = stereo.compute(img_gray1, img_gray2).astype(np.float32) / 16.0
        plt.imshow(disparity, 'gray')
        plt.show(block=True)