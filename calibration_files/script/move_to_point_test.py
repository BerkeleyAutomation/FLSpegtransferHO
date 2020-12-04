import numpy as np
from FLSpegtransfer.vision.BlockDetection3D import BlockDetection3D
from FLSpegtransfer.vision.ZividCapture import ZividCapture
from FLSpegtransfer.motion.dvrkDualArm import dvrkDualArm
from FLSpegtransfer.path import *
from FLSpegtransfer.motion.dvrkController import dvrkController

def move_origin(motion, which_arm):
    if which_arm == "PSM1":
        motion.set_arm_position(pos1=[0.055, 0.0, -0.13])
        motion.set_jaw(jaw1=np.deg2rad([5.0]))
    elif which_arm == "PSM2":
        motion.set_arm_position(pos2=[-0.055, 0.0, -0.13])
        motion.set_jaw(jaw2=np.deg2rad([0.0]))

def move_ready(motion, which_arm):
    if which_arm == "PSM1":
        motion.set_arm_position(pos1=[0.127, 0.0, -0.10])
    elif which_arm == "PSM2":
        motion.set_arm_position(pos2=[-0.127, 0.0, -0.10])

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
which_arm = 'PSM1'
Trc = np.load(root + '/calibration_files/Trc_' + which_camera + '_' + which_arm + '.npy')
Tpc = np.load(root + '/calibration_files/Tpc_' + which_camera + '.npy')
bd = BlockDetection3D(Tpc)
# dvrk = dvrkDualArm()
# dvrk = dvrkController(comp_hysteresis=False, stop_collision=False)
dvrk = dvrkDualArm()
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
            dvrk.set_pose(pos1=[pr[0], pr[1], pr[2] + 0.01], rot1=[0.0, 0.0, 0.0, 1.0])
        elif which_arm == 'PSM2':
            # dvrk.set_arm_position(pos2=[pr[0], pr[1], pr[2] + 0.005])
            dvrk.set_pose(pos2=[pr[0], pr[1], pr[2] + 0.01], rot2=[0.0, 0.0, 0.0, 1.0])

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