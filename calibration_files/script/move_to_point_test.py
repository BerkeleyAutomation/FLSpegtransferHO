import numpy as np
from FLSpegtransfer.vision.BlockDetection3D import BlockDetection3D
from FLSpegtransfer.vision.ZividCapture import ZividCapture
from FLSpegtransfer.motion.dvrkArm import dvrkArm
from FLSpegtransfer.motion.dvrkController import dvrkController
from FLSpegtransfer.path import *


def move_origin(obj, which_arm):
    if which_arm == "PSM1":
        obj.set_pose_interpolate(pos=[0.055, 0.0, -0.10], rot=[0.0, 0.0, 0.0, 1.0])
        obj.set_jaw_interpolate(jaw=np.deg2rad([5.0]))
    elif which_arm == "PSM2":
        obj.set_pose_interpolate(pos=[-0.055, 0.0, -0.10], rot=[0.0, 0.0, 0.0, 1.0])
        obj.set_jaw_interpolate(jaw=np.deg2rad([0.0]))

def move_ready(obj, which_arm):
    if which_arm == "PSM1":
        obj.set_pose_interpolate(pos=[0.127, 0.0, -0.10], rot=[0.0, 0.0, 0.0, 1.0])
    elif which_arm == "PSM2":
        obj.set_pose_interpolate(pos=[-0.127, 0.0, -0.10], rot=[0.0, 0.0, 0.0, 1.0])

def transform_task2robot(point, Tpc, Trc, inverse=False):
    point = np.array(point)
    if inverse == False:
        Tcp = np.linalg.inv(Tpc)  # (mm)
        Tcp[:3, -1] = Tcp[:3, -1] * 0.001  # (m)
        Trp = Trc.dot(Tcp)
        Rrp = Trp[:3, :3]
        trp = Trp[:3, -1]
        transformed = Rrp.dot(point * 0.001) + trp
    else:
        Tcr = np.linalg.inv(Trc)   # (m)
        Tcr[:3,-1] = Tcr[:3,-1] * 1000  # (mm)
        Tpr = Tpc.dot(Tcr)
        Rpr = Tpr[:3, :3]
        tpr = Tpr[:3, -1]
        transformed = Rpr.dot(point * 1000) + tpr
    return transformed

which_camera = 'inclined'
which_arm = 'PSM2'
dvrk = dvrkArm('/'+which_arm)
Trc = np.load(root + '/calibration_files/Trc_' + which_camera + '_' + which_arm + '.npy')
Tpc = np.load(root + '/calibration_files/Tpc_' + which_camera + '.npy')
bd = BlockDetection3D(Tpc)
# dvrk = dvrkController(comp_hysteresis=False, stop_collision=False)
zivid = ZividCapture(which_camera=which_camera)
zivid.start()
while True:
    move_origin(dvrk, which_arm)
    img_color, _, img_point = zivid.capture_3Dimage(color='BGR')
    bd.find_pegs(img_color, img_point)
    import pdb;    pdb.set_trace()
    for i in range(12):
        pr = transform_task2robot(bd.pnt_pegs[i], Trc=Trc, Tpc=Tpc)
        move_ready(dvrk, which_arm)
        if which_arm == 'PSM1':
            # dvrk.set_arm_position(pos1=[pr[0], pr[1], pr[2]+0.005])
            dvrk.set_pose_interpolate(pos=[pr[0], pr[1], pr[2] + 0.01], rot=[0.0, 0.0, 0.0, 1.0])
        elif which_arm == 'PSM2':
            # dvrk.set_arm_position(pos2=[pr[0], pr[1], pr[2] + 0.005])
            dvrk.set_pose_interpolate(pos=[pr[0], pr[1], pr[2] + 0.01], rot=[0.0, 0.0, 0.0, 1.0])

    # if not grasping_pose == [[]] * 2:
    #     Rrc = Trc[:3, :3]  # transform
    #     trc = Trc[:3, -1]
    #     for gp in grasping_pose:
    #         n, ang, Xc, Yc, Zc, seen = gp
    #         if seen == True:
    #             x, y, z = Rrc.dot([Xc, Yc, Zc]) + trc  # position in terms of the robot's coordinate
    #             gp1 = [x, y, z, ang]
    #             dvrk.set_pose(jaw1=[0.0])
    #             dvrk.set_arm_position(pos1=[0.110, 0.0, -0.105])
    #             dvrk.set_arm_position(pos1=[x,y,z+0.00])

    # cv2.imshow("color", img_pegs_ovl)
    # cv2.imshow("depth", img_blks_ovl)
    # cv2.waitKey(1)