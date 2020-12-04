import sys
for p in sys.path:
    if p == '/opt/ros/kinetic/lib/python2.7/dist-packages':
        sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
from FLSpegtransfer.vision.ZividCapture import ZividCapture
from FLSpegtransfer.vision.BlockDetection3D import BlockDetection3D
from FLSpegtransfer.vision.PegboardCalibration import PegboardCalibration
from FLSpegtransfer.vision.VisualizeDetection import VisualizeDetection
from FLSpegtransfer.vision.GraspingPose3D import GraspingPose3D


zivid = ZividCapture()
# zivid.start()
peg = PegboardCalibration()
bd = BlockDetection3D()
vd = VisualizeDetection()
gp = GraspingPose3D()

import time
st = time.time()
for i in range(1, 80):
    color = np.load('dropping_block_images/failed/'+str(i)+'/color_dropping.npy')
    point = np.load('dropping_block_images/failed/'+str(i)+'/point_dropping.npy')
    color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    # color, depth, point = zivid.capture_3Dimage(color='BGR')
    ycr, hcr, xcr, wcr = peg.define_boundary(color)
    dx = 200
    dy = 200
    zivid.ycr = ycr-dy
    zivid.hcr = hcr+2*dy
    zivid.xcr = xcr-dx
    zivid.wcr = wcr+2*dx
    print(i)
    T, pnt_blk, pnt_mask = bd.find_block_servo(color, point)
    # vd.plot3d(pnt_blocks=pnt_blk, pnt_masks=pnt_mask, pnt_pegs=bd.pnt_pegs)

print (time.time() - st)