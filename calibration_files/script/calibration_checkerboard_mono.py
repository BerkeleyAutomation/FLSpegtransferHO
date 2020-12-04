import cv2
import numpy as np

# grid information
row = 5
col = 8
length = 12.0  # mm
nb_image = 17

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((row * col, 3), np.float32)
objp[:, :2] = np.mgrid[0:col, 0:row].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

for i in range(nb_image):
    path = "/home/hwangmh/pycharmprojects/stereo_image_checkerboard/white_board/"
    filename = 'img_left'+str(int(i))+'.png'
    img = cv2.imread(path+filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print (filename)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (col, row), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        corners_ref = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        print ("Found corners.")
        print ("")

        # append points
        imgpoints.append(corners_ref)
        objpoints.append(objp)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (col, row), corners_ref, ret)
        cv2.imshow('img', img)
        cv2.waitKey(1)
    else:
        print ("Failed to find corners.")
        print("")

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("mtx: ", mtx)
print("dist: ", dist)
# print("rvecs: ", rvecs)
# print("tvecs: ", tvecs)

# Get new camera marix
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))   # if alpha=1, all pixels are maintained
print(newcameramtx)

# Rectify image
# option 1)
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]
cv2.imwrite('calibresult1.png', dst)

# option 2)
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]
cv2.imwrite('calibresult2.png', dst)