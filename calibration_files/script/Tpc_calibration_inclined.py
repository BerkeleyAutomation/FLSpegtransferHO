import cv2
import numpy as np
from FLSpegtransfer.vision.ZividCapture import ZividCapture
from FLSpegtransfer.vision.PegboardCalibration import PegboardCalibration

zivid = ZividCapture(which_camera='inclined')
zivid.start()
peg = PegboardCalibration()

# define region of interest
color, depth, point = zivid.capture_3Dimage(color='BGR')
ycr, hcr, xcr, wcr = peg.define_boundary(color)
dx = 200
dy = 200
zivid.ycr = ycr-dy
zivid.hcr = hcr+2*dy
zivid.xcr = xcr-dx
zivid.wcr = wcr+2*dx

# pegboard registration if necessary
peg.registration_pegboard(color, point)