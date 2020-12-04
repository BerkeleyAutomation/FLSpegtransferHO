import sys
for p in sys.path:
    if p == '/opt/ros/kinetic/lib/python2.7/dist-packages':
        sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
from FLSpegtransfer.vision.ZividCapture import ZividCapture
from FLSpegtransfer.vision.BallDetectionRGBD import BallDetectionRGBD
from FLSpegtransfer.utils.RANSAC.RANSAC_circle import RANSAC_circle
from sklearn.metrics import mean_squared_error

ball = BallDetectionRGBD()
ransac = RANSAC_circle()
zivid = ZividCapture()
# zivid.start()

root = '/home/hwangmh/pycharmprojects/FLSpegtransfer/'
Tpc = np.load(root + 'calibration_files/Tpc.npy')
hsv_lower_red = np.array([0 - 20, 130, 40])
hsv_upper_red = np.array([0 + 20, 255, 255])
# masking_depth = [-150, 0]
masking_depth = [500, 730]


def mask_image(img_color, img_point):
    # 2D color masking
    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
    img_masked = cv2.inRange(img_hsv, hsv_lower_red, hsv_upper_red)

    # noise filtering
    kernel = np.ones((2, 2), np.uint8)
    img_masked = cv2.morphologyEx(img_masked, cv2.MORPH_OPEN, kernel)

    # color masking
    con1 = (img_masked == 255)
    arg1 = np.argwhere(con1)
    pnt1 = img_point[con1]

    # remove nan
    con2 = (~np.isnan(pnt1).any(axis=1))
    arg2 = np.argwhere(con2)
    pnt2 = pnt1[con2]

    # transform w.r.t. task coordinate
    Tpc = np.eye(4)
    pnt2_tr = BallDetectionRGBD.transform(pnt2, Tpc)

    # depth masking
    con3 = (pnt2_tr[:, 2] > masking_depth[0]) & (pnt2_tr[:, 2] < masking_depth[1])
    arg3 = np.argwhere(con3)

    # creat mask where color & depth conditions hold
    arg_mask = np.squeeze(arg1[arg2[arg3]])
    mask = np.zeros_like(img_masked)
    mask[arg_mask[:, 0], arg_mask[:, 1]] = 255
    return mask


def find_balls(img_color, img_point, nb_balls, visualize=False):
    # mask color & depth
    masked = mask_image(img_color, img_point)

    if visualize:
        cv2.imshow("", masked)
        cv2.waitKey(0)

    # Find contours
    cnts, _ = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # fit spheres by contour points
    max_cnts = []
    best_models = []
    max_ratios = []
    for c in cnts:
        if len(c) < 50:
            continue
        else:
            color = np.copy(img_color)
            cv2.drawContours(color, [c], -1, (0, 255, 255), 1)

            if visualize:
                cv2.imshow("", color)
                cv2.waitKey(0)

            # Find ball in 2D image
            args = np.squeeze(c)
            x = args[:,0]
            y = args[:,1]
            best_model, max_cnt, max_ratio = ransac.fit(x, y, 100, threshold=2)
            best_models.append(best_model)
            max_cnts.append(max_cnt)
            max_ratios.append(max_ratio)

    # sort by best matched counts
    arg = np.argsort(max_ratios)[::-1]
    best_models = np.array(best_models)[arg][:nb_balls]

    # fit spheres by 3D points
    pb = []
    for model in best_models:
        xi,yi,ri = model
        infilled = np.zeros(np.shape(img_color), np.uint8)
        cv2.circle(infilled, (int(xi), int(yi)), int(ri), (0, 255, 255), -1)
        infilled = cv2.cvtColor(infilled, cv2.COLOR_BGR2GRAY)
        ball_masked = cv2.bitwise_and(masked, masked, mask=infilled)

        if visualize:
            cv2.imshow("", ball_masked)
            cv2.waitKey(0)

        # Get the point sets
        args = np.argwhere(ball_masked == 255)
        points_ball = img_point[args[:, 0], args[:, 1]]

        # Linear regression to fit the circle into the point cloud
        xc, yc, zc, rc = BallDetectionRGBD.fit_circle_3d(points_ball[:, 0], points_ball[:, 1], points_ball[:, 2])
        pb.append([xc, yc, zc, rc])

    # sort by x-position
    pb = np.array(pb)
    arg = np.argsort(pb[:, 0])
    return pb[arg]


def get_model_position(nb_exp):
    L1 = 30.0
    L2 = 50.0
    L3 = 70.0
    L4 = 90.0
    label_ZividM1 = np.array([[L2, L3, L4, L1],
                           [L4, L1, L3, L2],
                           [L2, L1, L4, L3],
                           [L2, L3, L1, L4],
                           [L3, L1, L4, L2],
                           [L3, L4, L1, L2],
                           [L4, L1, L2, L3],
                           [L2, L4, L1, L3],
                           [L4, L2, L1, L3],
                           [L3, L2, L1, L4],

                           [L3, L1, L4, L2],
                           [L4, L1, L2, L3],
                           [L3, L4, L1, L2],
                           [L3, L4, L1, L2],
                           [L4, L3, L1, L2],
                           [L4, L3, L1, L2],
                           [L3, L4, L1, L2],
                           [L2, L1, L3, L4],
                           [L3, L4, L1, L2],
                           [L4, L3, L1, L2]])
    label_ZividM2 = np.array([[[3.5, 0.5], [4.5, 3.5], [5.5, 0.5], [4.0, 0.0], [5.0, 3.0], [1.0, 3.0]],
                           [[1.5, 2.5], [2.5, 0.5], [3.5, 4.5], [3.0, 1.0], [1.0, 3.0], [4.0, 2.0]],
                           [[1.0, 2.0], [4.5, 2.5], [5.0, 0.0], [2.5, 1.5], [1.0, 3.0], [0.5, 2.5]],
                           [[5.0, 0.0], [2.5, 1.5], [4.0, 1.0], [1.5, 2.5], [4.0, 4.0], [2.5, 1.5]],
                           [[2.5, 0.5], [2.0, 0.0], [3.0, 4.0], [2.5, 1.5], [1.5, 2.5], [4.0, 1.0]],
                           [[2.0, 0.0], [2.5, 0.5], [2.0, 2.0], [2.5, 1.5], [2.0, 0.0], [1.5, 0.5]],
                           [[1.5, 2.5], [4.0, 2.0], [2.0, 4.0], [0.5, 1.5], [2.5, 0.5], [2.0, 2.0]],
                           [[4.0, 1.0], [2.0, 1.0], [5.0, 0.0], [3.0, 1.0], [4.0, 4.0], [3.0, 1.0]],
                           [[1.0, 4.0], [5.5, 1.5], [4.0, 4.0], [2.5, 1.5], [0.0, 5.0], [1.5, 2.5]],
                           [[5.0, 0.0], [1.5, 2.5], [4.0, 4.0], [1.5, 2.5], [1.0, 4.0], [2.5, 1.5]],

                           [[1.5, 2.5], [4.0, 4.0], [5.0, 0.0], [2.5, 1.5], [2.5, 1.5], [1.0, 4.0]],
                           [[1.5, 2.5], [4.5, 2.5], [2.5, 0.5], [2.0, 1.0], [1.0, 3.0], [5.0, 0.0]],
                           [[2.0, 0.0], [2.5, 0.5], [3.0, 4.0], [2.5, 1.5], [1.0, 4.0], [1.5, 2.5]],
                           [[2.0, 0.0], [2.5, 0.5], [2.0, 2.0], [2.5, 1.5], [2.0, 0.0], [0.5, 1.5]],
                           [[4.0, 0.0], [1.0, 3.0], [3.5, 4.5], [5.0, 3.0], [0.5, 3.5], [0.5, 5.5]],
                           [[3.0, 1.0], [1.0, 1.0], [4.0, 2.0], [2.0, 2.0], [3.0, 1.0], [3.0, 1.0]],
                           [[0.5, 2.5], [1.0, 3.0], [4.0, 2.0], [1.5, 2.5], [3.5, 4.5], [3.0, 1.0]],
                           [[1.5, 2.5], [5.0, 0.0], [1.0, 4.0], [2.5, 1.5], [1.5, 5.5], [4.0, 4.0]],
                           [[4.0, 2.0], [2.5, 0.5], [2.0, 2.0], [1.5, 2.5], [4.0, 2.0], [1.5, 0.5]],
                           [[4.0, 0.0], [1.0, 3.0], [3.5, 4.5], [5.0, 3.0], [0.5, 3.5], [0.5, 5.5]]])
    # label_ZividS1 = np.array([[L2, L3, L4, L1],
    #                           [L2, L1, L3, L4],
    #                           [L2, L1, L3, L4],
    #                           [L2, L1, L3, L4],
    #                           [L2, L3, L1, L4],
    #                           [L2, L1, L3, L4],
    #                           [L2, L1, L4, L3],
    #                           [L2, L1, L3, L4],
    #                           [L2, L4, L1, L3],
    #                           [L4, L1, L3, L2],
    #
    #                           [L1, L4, L3, L2],
    #                           [L3, L4, L2, L1],
    #                           [L3, L4, L2, L1],
    #                           [L3, L4, L2, L1],
    #                           [L3, L4, L2, L1],
    #                           [L3, L2, L4, L1],
    #                           [L3, L2, L4, L1],
    #                           [L2, L1, L3, L4],
    #                           [L2, L1, L4, L3],
    #                           [L2, L1, L3, L4]])
    # label_ZividS2 = np.array([[[3.5, 0.5], [2.0, 4.0], [5.5, 0.5], [3.5, 1.5], [5.0, 3.0], [1.5, 1.5]],
    #                           [[2.5, 2.5], [5.0, 0.0], [2.5, 5.5], [2.5, 2.5], [0.0, 3.0], [0.5, 2.5]],
    #                           [[2.5, 0.5], [5.0, 2.0], [0.5, 5.5], [2.5, 2.5], [0.0, 3.0], [0.5, 2.5]],
    #                           [[2.5, 1.5], [5.0, 2.0], [5.5, 0.5], [0.5, 2.5], [3.0, 2.0], [2.5, 0.5]],
    #                           [[2.5, 0.5], [2.5, 1.5], [0.5, 5.5], [1.0, 2.0], [3.0, 5.0], [3.0, 2.0]],
    #                           [[2.0, 0.0], [4.0, 0.0], [5.5, 0.5], [2.0, 0.0], [3.5, 0.5], [1.5, 0.5]],
    #                           [[5.5, 0.5], [4.0, 2.0], [3.5, 0.5], [1.5, 1.5], [5.0, 3.0], [3.5, 1.5]],
    #                           [[2.0, 0.0], [3.5, 3.5], [6.0, 0.0], [1.5, 3.5], [4.0, 0.0], [3.5, 2.5]],
    #                           [[1.0, 3.0], [2.0, 0.0], [3.5, 3.5], [1.0, 3.0], [0.5, 2.5], [1.5, 3.5]],
    #                           [[1.0, 3.0], [2.5, 0.5], [5.5, 3.5], [1.5, 3.5], [4.5, 0.5], [3.0, 4.0]],
    #
    #                           [[3.5, 0.5], [4.0, 3.0], [6.0, 0.0], [2.5, 0.5], [5.5, 3.5], [3.0, 4.0]],
    #                           [[1.5, 2.5], [5.0, 3.0], [5.0, 3.0], [5.5, 3.5], [3.5, 0.5], [6.0, 0.0]],
    #                           [[2.5, 0.5], [5.0, 3.0], [5.0, 3.0], [2.5, 2.5], [2.5, 3.5], [6.0, 0.0]],
    #                           [[2.5, 0.5], [3.5, 1.5], [3.0, 5.0], [2.0, 1.0], [3.5, 2.5], [1.5, 1.5]],
    #                           [[2.0, 3.0], [1.5, 3.5], [3.0, 5.0], [1.5, 1.5], [3.0, 0.0], [1.5, 1.5]],
    #                           [[3.5, 0.5], [2.0, 3.0], [5.0, 3.0], [2.5, 0.5], [5.5, 0.5], [3.0, 0.0]],
    #                           [[3.5, 0.5], [3.5, 1.5], [5.0, 3.0], [4.0, 2.0], [0.5, 5.5], [1.5, 1.5]],
    #                           [[2.0, 0.0], [4.0, 0.0], [6.0, 0.0], [2.0, 0.0], [4.0, 0.0], [2.0, 0.0]],
    #                           [[2.5, 2.5], [3.0, 2.0], [5.0, 0.0], [4.5, 0.5], [2.5, 2.5], [2.0, 2.0]],
    #                           [[5.5, 0.5], [0.5, 3.5], [3.5, 4.5], [5.0, 3.0], [3.0, 1.0], [4.0, 0.0]]])

    dim1 = label_ZividM1[nb_exp - 1]
    dim2 = label_ZividM2[nb_exp - 1]
    # dim1 = label_ZividS1[nb_exp-1]
    # dim2 = label_ZividS2[nb_exp-1]
    dist = lambda x,y,z:np.sqrt((20.0*x)**2 + (20.0*y)**2 + z**2)
    D12 = dist(dim2[0][0], dim2[0][1], dim1[0] - dim1[1])
    D13 = dist(dim2[1][0], dim2[1][1], dim1[0] - dim1[2])
    D14 = dist(dim2[2][0], dim2[2][1], dim1[0] - dim1[3])
    D23 = dist(dim2[3][0], dim2[3][1], dim1[1] - dim1[2])
    D24 = dist(dim2[4][0], dim2[4][1], dim1[1] - dim1[3])
    D34 = dist(dim2[5][0], dim2[5][1], dim1[2] - dim1[3])
    return [D12, D13, D14, D23, D24, D34]


def get_measured_position(pbs):
    D12 = np.linalg.norm([pbs[0][:3] - pbs[1][:3]])
    D13 = np.linalg.norm([pbs[0][:3] - pbs[2][:3]])
    D14 = np.linalg.norm([pbs[0][:3] - pbs[3][:3]])
    D23 = np.linalg.norm([pbs[1][:3] - pbs[2][:3]])
    D24 = np.linalg.norm([pbs[1][:3] - pbs[3][:3]])
    D34 = np.linalg.norm([pbs[2][:3] - pbs[3][:3]])
    return [D12, D13, D14, D23, D24, D34]


total_modeled = []
total_measured = []
for nb_exp in range(1, 21):
    # load image
    print ("number: ", nb_exp)
    color = np.load('data/conf' + str(nb_exp) + '/color.npy')
    point = np.load('data/conf' + str(nb_exp) + '/point.npy')
    # color, _, point = zivid.capture_3Dimage(color='BGR')

    # measure positions from images
    # Find balls
    pbs = find_balls(color, point, nb_balls=4, visualize=False)
    overlayed = ball.overlay_ball(color, pbs)
    measured = get_measured_position(pbs)

    # get positions from model
    modeled = get_model_position(nb_exp)

    # print out results
    print ("Modeled: ", modeled)
    print ("Measured: ", measured)
    modeled = np.array(modeled)
    measured = np.array(measured)
    print ("Error: ", modeled-measured)

    RMSE = np.sqrt(mean_squared_error(modeled, measured))
    SD = np.std(modeled - measured)
    print("RMSE=", RMSE, '(mm)')
    print("SD=", SD)

    # visualize
    # cv2.imshow("", overlayed)
    # cv2.waitKey(0)
    cv2.imwrite('data/conf' + str(nb_exp) + '/color_detected.png', overlayed)

    total_modeled.append(modeled)
    total_measured.append(measured)

# save the data
np.save("total_modeled", total_modeled)
np.save("total_measured", total_measured)