import cv2
print (cv2.__version__)
import numpy as np
import FLSpegtransfer.utils.CmnUtil as U
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


def collect_checkerboard_corners(row, col, nb_image, grid_size):
    # prepare object points, like ((0,0,0), (1,0,0), (2,0,0) ....,(row,col,0))*grid_size
    objp = np.zeros((row * col, 3), np.float32)
    objp[:, :2] = np.mgrid[0:col, 0:row].T.reshape(-1, 2)
    objp = objp*grid_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints1 = []  # 2d points in image plane.
    imgpoints2 = []
    for i in range(nb_image):
        # path = "/home/hwangmh/pycharmprojects/stereo_image_checkerboard/white_board/"
        path = "calibration_alliedvision/"
        filename_left = 'img_left' + str(int(i)) + '.png'
        filename_right = 'img_right' + str(int(i)) + '.png'
        img_left = cv2.imread(path+filename_left)
        img_right = cv2.imread(path+filename_right)
        if img_left is None or img_right is None or len(img_left) == 0 or len(img_right) == 0:
            pass
        else:
            gray1 = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret1, corners1 = cv2.findChessboardCorners(gray1, (col, row), flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FILTER_QUADS)
            ret2, corners2 = cv2.findChessboardCorners(gray2, (col, row), flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FILTER_QUADS)

            if ret1 and ret2:
                # If found, add object points, image points (after refining them)
                # termination criteria
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_ref1 = cv2.cornerSubPix(gray1, corners1, (5, 5), (-1, -1), criteria)
                corners_ref2 = cv2.cornerSubPix(gray2, corners2, (5, 5), (-1, -1), criteria)
                objpoints.append(objp)
                imgpoints1.append(corners_ref1)
                imgpoints2.append(corners_ref2)
                print("image " + str(i) + " completed.")
    return objpoints, imgpoints1, imgpoints2


def collect_trans_rot_vectors(objpoints, imgpoints1, imgpoints2, K, D):
    # obtain R, T matrices from left cam to right cam
    Rs = []
    Ts = []
    for objp, corners_ref1, corners_ref2 in zip(objpoints, imgpoints1, imgpoints2):
        # Transformation from model to the detected
        _, rvecs1, tvecs1, _ = cv2.solvePnPRansac(objp, corners_ref1, K[0], D[0])
        _, rvecs2, tvecs2, _ = cv2.solvePnPRansac(objp, corners_ref2, K[1], D[1])
        tc1m = tvecs1
        tc2m = tvecs2
        Rc1m = cv2.Rodrigues(rvecs1)[0]
        Rc2m = cv2.Rodrigues(rvecs2)[0]

        # Transformation matrix from cam 1 to cam 2
        Rc1c2 = Rc1m.dot(Rc2m.T)
        tc1c2 = Rc1m.dot(-Rc2m.T.dot(tc2m)) + tc1m
        Rs.append(Rc1c2)
        Ts.append(np.squeeze(tc1c2))
    Rs = np.array(Rs)
    Ts = np.array(Ts)
    return Rs, Ts


def plot_trans_rot_vectors(Rs, Ts):
    Eulers = np.array([U.R_to_euler(R) for R in Rs])
    Ts = np.array(Ts)

    fig = plt.figure()
    ax = fig.add_subplot(211, projection='3d')  # translation
    ax.scatter(Ts[:, 0], Ts[:, 1], Ts[:, 2])
    ax.set_xlabel('X Label (m)')
    ax.set_ylabel('Y Label (m)')
    ax.set_zlabel('Z Label (m)')

    ax2 = fig.add_subplot(212, projection='3d')  # rotation
    Eulers_deg = np.rad2deg(Eulers)  # (deg)
    ax2.scatter(Eulers_deg[:, 0], Eulers_deg[:, 1], Eulers_deg[:, 2])
    ax2.set_xlabel('X Label (deg)')
    ax2.set_ylabel('Y Label (deg)')
    ax2.set_zlabel('Z Label (deg)')
    plt.show()


def remove_outliers(Rs, Ts):
    Eulers = np.array([U.R_to_euler(R) for R in Rs])
    Ts = np.array(Ts)

    # for translation
    clustering = DBSCAN(eps=0.3, min_samples=2).fit(Ts)
    index = clustering.labels_

    # for rotation
    clustering2 = DBSCAN(eps=0.7, min_samples=2).fit(Eulers)
    index2 = clustering2.labels_

    # cross sectional index
    index_cross = index + index2
    Ts = Ts[index_cross == 0]
    T = np.average(Ts, axis=0)
    Euler = Eulers[index_cross == 0]
    Euler = np.average(Euler, axis=0)
    R = U.euler_to_R(Euler)
    return R, T


img_shape = (1280, 960)

# collect checkerboard corners
objpoints, imgpoints1, imgpoints2 = collect_checkerboard_corners(row=11, col=7, nb_image=100, grid_size=6.85*0.001)
# get camera intrinsics
D = np.zeros((2,5))
K = np.zeros((2,3,3))
flag = 0
flag |= cv2.CALIB_FIX_K4
flag |= cv2.CALIB_FIX_K5
ret1, K[0], D[0], rvecs1, tvecs1 = cv2.calibrateCamera(objpoints, imgpoints1, img_shape, None, None, flags=flag)
ret2, K[1], D[1], rvecs2, tvecs2 = cv2.calibrateCamera(objpoints, imgpoints2, img_shape, None, None, flags=flag)
print ("")
print ("RMS errors of calibration: ", ret1, ret2)
print ("Camera intrinsics")
print ("K=", K)
print ("D=", D)


#######################################
##  Remove outliers (Manually delete files)
#######################################

# collect trans & rot vectors between two cameras (from right to left)
Rs, Ts = collect_trans_rot_vectors(objpoints, imgpoints1, imgpoints2, K, D)

# plot trans & rot vectors collected
plot_trans_rot_vectors(Rs, Ts)

# remove outliers
# R, T = remove_outliers(Rs, Ts)

#######################################
##   Stereo calibration
#######################################
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC
# flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
# flags |= cv2.CALIB_USE_INTRINSIC_GUESS
# flags |= cv2.CALIB_FIX_FOCAL_LENGTH
# flags |= cv2.CALIB_FIX_ASPECT_RATIO
# flags |= cv2.CALIB_ZERO_TANGENT_DIST
# flags |= cv2.CALIB_RATIONAL_MODEL
# flags |= cv2.CALIB_SAME_FOCAL_LENGTH
# flags |= cv2.CALIB_FIX_K3
# flags |= cv2.CALIB_FIX_K4
# flags |= cv2.CALIB_FIX_K5
stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1, 1e-5)
ret, K[0], D[0], K[1], D[1], R, T, E, F = \
    cv2.stereoCalibrate(objpoints, imgpoints1, imgpoints2, K[0], D[0], K[1], D[1], img_shape)
    # cv2.stereoCalibrate(objpoints, imgpoints1, imgpoints2, K[0], D[0], K[1], D[1], img_shape,
    #                     criteria=stereocalib_criteria, flags=flags)

print("")
print("RMS errors of stereo calibration: ", ret1, ret2)
print("New camera intrinsics")
print("K=", K)
print("D=", D)


#######################################
##   Rectification
#######################################
img_shape = (1280, 960)
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K[0], D[0], K[1], D[1], img_shape, R, T)
mapx1, mapy1 = cv2.initUndistortRectifyMap(K[0], D[0], R1, P1, img_shape, cv2.CV_32F)
mapx2, mapy2 = cv2.initUndistortRectifyMap(K[1], D[1], R2, P2, img_shape, cv2.CV_32F)

# print results
print ("Stereo calibration results")
print("R=", R)
Euler = U.R_to_euler(R)
print("Euler=", np.rad2deg(Euler), "(deg)")
print ("T=", np.array(T), "(m)")
print ("E=", E)
print ("F=", F)
print ("")
print ("Rectify matrices")
print ("R1=", R1)
print ("R2=", R2)
print ("P1=", P1)
print ("P2=", P2)
print ("Q=", Q)
print ("mapx1=", mapx1)
print ("mapy1=", mapy1)
print ("mapx2=", mapx2)
print ("mapy2=", mapy2)


#######################################
##   Save matrices
#######################################
# intrinsics
np.save("K", K)
np.save("D", D)
np.save("R", R)
np.save("T", T)
np.save("E", E)
np.save("F", F)

# rectification
np.save("R1", R1)
np.save("R2", R2)
np.save("P1", P1)
np.save("P2", P2)
np.save("Q", Q)
np.save("mapx1", mapx1)
np.save("mapy1", mapy1)
np.save("mapx2", mapx2)
np.save("mapy2", mapy2)
print ("saved!")



