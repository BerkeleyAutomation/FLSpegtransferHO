from FLSpegtransfer.motion.dvrkArm import dvrkArm
from FLSpegtransfer.vision.ZividCapture import ZividCapture
import threading

arm1 = dvrkArm('/PSM1')
arm2 = dvrkArm('/PSM2')
zivid = ZividCapture()
zivid.start()

def run():
    while True:
        arm1.set_pose(pos=[0.05, 0.0, -0.15], rot=[0.0, 0.0, 0.0, 1.0])
        zivid.capture_3Dimage()
        arm1.set_pose(pos=[-0.05, 0.0, -0.15], rot=[0.0, 0.0, 0.0, 1.0])
        zivid.capture_3Dimage()
        arm1.set_pose(pos=[0.05, 0.0, -0.15], rot=[0.0, 0.0, 0.0, 1.0])
        zivid.capture_3Dimage()
        arm1.set_pose(pos=[-0.05, 0.0, -0.15], rot=[0.0, 0.0, 0.0, 1.0])
        zivid.capture_3Dimage()
        arm1.set_pose(pos=[0.05, 0.0, -0.15], rot=[0.0, 0.0, 0.0, 1.0])
        zivid.capture_3Dimage()
        arm1.set_pose(pos=[-0.05, 0.0, -0.15], rot=[0.0, 0.0, 0.0, 1.0])
        print ("completed_arm1")

th = threading.Thread(target=run)
th.start()

while True:
    arm2.set_pose(pos=[0.05, 0.0, -0.15], rot=[0.0, 0.0, 0.0, 1.0])
    zivid.capture_3Dimage()
    arm2.set_pose(pos=[-0.05, 0.0, -0.15], rot=[0.0, 0.0, 0.0, 1.0])
    zivid.capture_3Dimage()
    arm2.set_pose(pos=[0.05, 0.0, -0.15], rot=[0.0, 0.0, 0.0, 1.0])
    zivid.capture_3Dimage()
    arm2.set_pose(pos=[-0.05, 0.0, -0.15], rot=[0.0, 0.0, 0.0, 1.0])
    zivid.capture_3Dimage()
    arm2.set_pose(pos=[0.05, 0.0, -0.15], rot=[0.0, 0.0, 0.0, 1.0])
    zivid.capture_3Dimage()
    arm2.set_pose(pos=[-0.05, 0.0, -0.15], rot=[0.0, 0.0, 0.0, 1.0])
    print ("completed_arm2")