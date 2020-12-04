import sys
for p in sys.path:
    if p == '/opt/ros/kinetic/lib/python2.7/dist-packages':
        sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import numpy as np
from FLSpegtransfer.vision.ZividCapture import ZividCapture
from FLSpegtransfer.vision.BallDetection import BallDetection
from FLSpegtransfer.vision.BlockDetection3D import BlockDetection3D
from FLSpegtransfer.motion.deprecated.dvrkMotionBridgeP import dvrkMotionBridgeP
from FLSpegtransfer.vision.PegboardCalibration import PegboardCalibration

BD = BallDetection()
block = BlockDetection3D()
zivid = ZividCapture()
zivid.start()
dvrk = dvrkMotionBridgeP()
peg = PegboardCalibration()
Trc = np.load('../calibration_files/Trc.npy')

def transform_task2robot(point, delta=False):
    point = np.array(point)
    Tcp = np.linalg.inv(block.Tpc)  # (mm)
    Tcp[:3, -1] = Tcp[:3, -1] * 0.001  # (m)
    Trp = Trc.dot(Tcp)
    Rrp = Trp[:3, :3]
    trp = Trp[:3, -1]
    if delta == False:
        transformed = Rrp.dot(point * 0.001) + trp
    else:
        transformed = Rrp.dot(point * 0.001)
    return transformed

def move_origin():

    jaw_org1 = [0.0]


# define ROI
pos_org1 = [0.080, 0.0, -0.095]
dvrk.set_arm_position(pos1=pos_org1)
color, _, point = zivid.capture_3Dimage(color='BGR')
ycr, hcr, xcr, wcr = peg.define_boundary(color)
dx = 200    # clearance of the region of interest
dy = 200
zivid.ycr = ycr - dy
zivid.hcr = hcr + 2 * dy
zivid.xcr = xcr - dx
zivid.wcr = wcr + 2 * dx

while True:
    dvrk.set_arm_position(pos1=[0.09, 0.0, -0.11])
    color, depth, point = zivid.capture_3Dimage(img_crop=True, color='BGR')
    color_org = np.copy(color)
    if color == [] or depth == [] or point == []:
        pass
    else:
        # Find peg points in 3D
        block.find_pegs(color, point)

        # Command robot to go above each peg points
        for p in block.pnt_pegs:
            pr = transform_task2robot(p)
            dvrk.set_pose(jaw1=[0.0])
            dvrk.set_arm_position(pos1=[0.130, 0.0, -0.105])
            dvrk.set_arm_position(pos1=[pr[0], pr[1], pr[2]+0.003])
            #
            # color, depth, point = zivid.capture_3Dimage(img_crop=True, color='BGR')
            #
            # # Find balls and overlay
            # pbs = BD.find_balls(color, depth, point)
            # color = BD.overlay_balls(color, pbs)
            #
            # print (pbs)
            # cv2.imshow("", color)
            # cv2.waitKey(0)
            #
            # # Find tool position, joint angles, and overlay
            # if pbs[0]==[] or pbs[1]==[]:
            #     pass
            # else:
            #     # Find tool position, joint angles, and overlay
            #     pt_act = BD.find_tool_position(pbs[0], pbs[1])    # measured tool position
            #     pt_act = np.array(pt_act) * 0.001  # (m)
            #     pt_act = BD.Rrc.dot(pt_act) + BD.trc            # convert it to robot's coordinate
            #     error = [pr[0], pr[1], pr[2]] - pt_act
            #     print (error)
            #
            #     # update robot's position
            #     # px_new, py_new, pz_new = [px, py, pz] + error
            #     # dvrk.set_pose(jaw1=[0.0])
            #     # dvrk.set_arm_position(pos1=[px_new, py_new, pz_new + 0.003])
            #
            #     # for display
            #     q0, q2, q3 = BD.ik_position(pt_act)
            #     # print(q0*180/np.pi, q2*180/np.pi, q3)
            #     color = BD.overlay_tool_position(color, [q0,q2,q3], (0,255,0))
            #
            # # cv2.imshow("images", img_color)
            # # cv2.waitKey(1)